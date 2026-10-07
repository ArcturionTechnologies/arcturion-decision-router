"""Bounded cascade: deterministic -> local -> primary -> selective judge -> reasoning."""
from __future__ import annotations
from dataclasses import asdict
import math
import time
from .protocol import Request, Judgment, Primitive, DecisionError, encode, digest, finite

class Engine:
    def __init__(self, providers, adapters, ledger, *, policy=None, max_calls=3,
                 max_seconds=60, max_cost_usd=.05, context_bytes=12000, escalation_bytes=6000,
                 coworker_providers=()):
        if not 1 <= max_calls <= 4 or not 1 <= max_seconds <= 120 or not finite(max_cost_usd,0,1):
            raise DecisionError("INVALID_BUDGET")
        self.providers,self.adapters,self.ledger = providers,adapters,ledger
        self.policy = policy or (lambda request: None)
        self.max_calls,self.max_seconds,self.max_cost_usd = max_calls,max_seconds,max_cost_usd
        self.context_bytes,self.escalation_bytes = context_bytes,escalation_bytes
        if not isinstance(coworker_providers, (list, tuple)) or not all(isinstance(n, str) for n in coworker_providers):
            raise DecisionError("INVALID_COWORKER_CONFIGURATION")
        self.coworker_providers = set(coworker_providers)

    def decide(self, state, question, options=None, decision_type="general", stakes="low",
               required_confidence=None, constraints=None, **kwargs):
        return self.evaluate(Request(state,question,options,decision_type,stakes,
                                     required_confidence,constraints or {},**kwargs))

    def score(self, state, question, levels, **kwargs):
        return self.decide(state,question,options=levels,primitive="SCORE",**kwargs)

    def rank(self, state, question, options, **kwargs):
        return self.decide(state,question,options=options,primitive="RANK",**kwargs)

    def gate(self, state, question, **kwargs):
        return self.decide(state,question,primitive="BOOLEAN_PROBABILITY",**kwargs)

    def evaluate(self, request):
        from .protocol import build_context
        from .consultation import normalize
        classification = normalize(request)
        start = time.monotonic()
        state_hash = digest(request.state)
        try:
            policy_block = self.policy(request)
            policy_error = False
        except Exception:
            policy_block, policy_error = True, True
        policy_block = policy_block or request.constraints.get("blocked") is True or request.constraints.get("approval_required") is True
        replay = self.ledger.reserve(request,state_hash)
        if replay:
            if policy_block:
                return dict(replay,decision=None,abstain=True,
                            reason="POLICY_UNAVAILABLE" if policy_error else "POLICY_BLOCKED",
                            route=replay["route"]+["T0:current_policy"])
            return replay
        result = {"schema_version":1,"decision_id":request.decision_id,
                  "requesting_agent":request.requesting_agent.upper(),"capacity":request.capacity,
                  "decision_type":request.decision_type,"stakes":request.stakes,"state_hash":state_hash,
                  "request_hash":digest(asdict(request)), "consultation":classification,
                  "decision":None,"confidence":0.0,"confidence_kind":"none","scores":{},
                  "abstain":True,"risk":request.stakes,"models_used":[],
                  "agreement":None,"escalated":False,"reason":"ESCALATION_REQUIRED",
                  "route":[],"provider_results":[],"escalation_reason":None,
                  "authorized":False,"advisory_only":True,"missing_information":[],
                  "usage":{"provider_calls":0,"input_tokens":0,"output_tokens":0,
                           "known_estimated_cost_usd":0.0,"unknown_cost_calls":0,"unknown_usage_calls":0}}
        def finish(judgment=None, reason=None):
            if judgment:
                result.update(decision=judgment.decision,confidence=judgment.confidence,
                              confidence_kind=judgment.confidence_kind,scores=judgment.scores,
                              abstain=judgment.abstain,reason=judgment.reason)
            if reason:
                result.update(decision=None,abstain=True,reason=reason)
            if self.coworker_providers:
                result["coworkers"] = {name: next(
                    ({"status": "unavailable" if r["abstain"] else "opinion_recorded",
                      "reason": r["reason"]} for r in result["provider_results"] if r["provider"] == name),
                    {"status": "not_called", "reason": "NOT_ELIGIBLE_OR_NOT_REQUIRED"})
                    for name in sorted(self.coworker_providers)}
            result["latency_ms"] = round((time.monotonic()-start)*1000,2)
            try:
                self.ledger.finish(request,result)
            except Exception:
                raise DecisionError("LEDGER_WRITE_FAILED") from None
            return result

        # Trusted policy callback is configured by the host; model outputs cannot edit it.
        if policy_error:
            return finish(reason="POLICY_UNAVAILABLE")
        if policy_block:
            result["route"].append("T0:policy")
            return finish(reason="POLICY_BLOCKED")
        if classification["exempt"]:
            result["route"].append("T0:consultation_exempt")
            return finish(reason="CONSULTATION_NOT_REQUIRED")
        required = request.constraints.get("required_fields",[])
        if not isinstance(required,list) or not all(isinstance(x,str) for x in required):
            return finish(reason="INVALID_REQUIRED_FIELDS")
        missing = [key for key in required if request.state.get(key) in (None,"",[],{})]
        if missing:
            result["missing_information"] = missing
            return finish(reason="INSUFFICIENT_INFORMATION")
        try:
            packet = build_context(request,self.context_bytes)
        except DecisionError as exc:
            result["missing_information"] = ["Supply compact decision-relevant state; select relevant_fields explicitly."]
            return finish(reason=str(exc).split(":")[0])
        result["context_bytes"] = len(encode(packet))
        try:
            deterministic = self.deterministic(request,packet)
        except (DecisionError,KeyError,TypeError,ValueError):
            return finish(reason="INVALID_DETERMINISTIC_RULE")
        if deterministic:
            result["route"].append("T0:deterministic")
            return finish(deterministic.validate(request))
        if request.primitive == Primitive.ABSTAIN:
            return finish(reason="INSUFFICIENT_INFORMATION")

        threshold = request.required_confidence
        if threshold is None:
            threshold = {"low":.8,"moderate":.85,"high":.9}[request.stakes]
        if request.stakes == "high":
            threshold = max(.9, threshold)
        attempted = set()
        judgments = []
        reserved_cost = 0.0

        def eligible(spec, tiers):
            return (spec.tier in tiers and spec.enabled and spec.availability not in {"disabled","unavailable","candidate"}
                    and spec.name in self.adapters and spec.name not in attempted
                    and request.primitive in spec.primitives and len(encode(packet)) <= spec.context_bytes)

        def choose(tiers, independent_of=None, *, primary=False):
            choices = [s for s in self.providers if eligible(s,tiers)
                       and (not primary or s.name not in self.coworker_providers)]
            if independent_of:
                groups = {s.independent_group or s.name for s,_ in independent_of}
                choices = [s for s in choices if (s.independent_group or s.name) not in groups]
            # Within an allowed tier: suitable specialty, then explicit estimated price.
            return min(choices,key=lambda s:(request.decision_type not in s.specialties,
                        s.input_usd_per_million if s.input_usd_per_million is not None else math.inf,
                        s.latency_ms or math.inf,s.name),default=None)

        def call(spec, call_packet):
            nonlocal reserved_cost
            if spec is None:
                return None
            if len(attempted) >= self.max_calls:
                result["route"].append("BUDGET:call_limit")
                return None
            remaining = self.max_seconds-(time.monotonic()-start)
            if remaining < 1:
                result["route"].append("BUDGET:deadline")
                return None
            if len(encode(call_packet)) > spec.context_bytes:
                result["route"].append("BUDGET:provider_context")
                return None
            # UTF-8 byte count is a conservative input token bound, plus schema overhead.
            if spec.local:
                upper_cost = 0.0
            elif spec.input_usd_per_million is None or spec.output_usd_per_million is None:
                result["route"].append("BUDGET:unknown_price")
                return None
            else:
                multiplier = len(request.options) if request.primitive == Primitive.RANK else 1
                upper_cost = ((len(encode(call_packet))*multiplier+4096)*spec.input_usd_per_million
                              +512*spec.output_usd_per_million)/1e6
            if reserved_cost + upper_cost > self.max_cost_usd:
                result["route"].append("BUDGET:cost_limit")
                return None
            reserved_cost += upper_cost
            attempted.add(spec.name)
            result["models_used"].append(spec.name)
            result["route"].append("T"+str(spec.tier)+":"+spec.name)
            at = time.monotonic()
            try:
                judgment = self.adapters[spec.name](request,call_packet,remaining).validate(request)
            except Exception:
                # Never expose provider exception text, credentials, or raw payloads.
                judgment = Judgment(abstain=True,reason="PROVIDER_ERROR",cost_usd=None)
            calibration = spec.calibration.get(request.decision_type,{})
            if calibration.get("samples",0) >= 30 and finite(calibration.get("confidence_cap")):
                judgment.confidence = min(judgment.confidence,calibration["confidence_cap"])
            record = dict(asdict(judgment),provider=spec.name,model=spec.model,tier=spec.tier,
                          role="coworker" if spec.name in self.coworker_providers else "primary_or_escalation",
                          latency_ms=round((time.monotonic()-at)*1000,2))
            result["provider_results"].append(record)
            usage = result["usage"]
            if judgment.provider_called:
                usage["provider_calls"] += 1
                usage["input_tokens"] += judgment.input_tokens or 0
                usage["output_tokens"] += judgment.output_tokens or 0
                usage["known_estimated_cost_usd"] += judgment.cost_usd or 0
                usage["unknown_cost_calls"] += int(judgment.cost_usd is None)
                usage["unknown_usage_calls"] += int(judgment.input_tokens is None or judgment.output_tokens is None)
            return judgment

        trigger = "DELIBERATION_REQUIRED" if request.requires_deliberation else None
        force_judges = request.multi_judge or request.stakes == "high"
        if not trigger:
            coworker_available = any(s.name in self.coworker_providers and eligible(s, {1,2,3}) for s in self.providers)
            first = choose({1}, primary=True) if request.stakes == "low" else None
            first = first or choose({2}, primary=True)
            value = call(first,packet)
            if value and not value.abstain:
                judgments.append((first,value))
                if value.confidence >= threshold and not force_judges and not coworker_available:
                    return finish(value)
            if first and first.tier == 1:
                primary = choose({2}, primary=True)
                value = call(primary,packet)
                if value and not value.abstain:
                    judgments.append((primary,value))
            # At most two useful independent judges. No all-provider fan-out.
            independent_groups = {s.independent_group or s.name for s,_ in judgments}
            if len(independent_groups) < 2 and (coworker_available or force_judges or not judgments or judgments[-1][1].confidence < threshold):
                second = choose({3},judgments) or choose({2,1},judgments)
                value = call(second,packet)
                if value and not value.abstain:
                    judgments.append((second,value))
            if judgments:
                winner,agreement,unstable = self.stability(request,judgments)
                result["agreement"] = agreement
                independent_groups = {s.independent_group or s.name for s,_ in judgments}
                primary_present = any(s.name not in self.coworker_providers for s,_ in judgments)
                if primary_present and not unstable and winner.confidence >= threshold and (not force_judges or len(independent_groups)>=2):
                    return finish(winner)
                trigger = "DECISION_UNSTABLE" if unstable else "LOW_CONFIDENCE"
                if force_judges and len(independent_groups)<2:
                    trigger = "INDEPENDENT_JUDGE_UNAVAILABLE"
                if not primary_present:
                    trigger = "PRIMARY_JUDGE_UNAVAILABLE"
            else:
                trigger = "PROVIDER_UNAVAILABLE"
        result["escalation_reason"] = trigger
        result["route"].append(trigger)
        result["handoff"] = {
            "owner": request.requesting_agent.upper(), "capacity": request.capacity,
            "kind": "responsible_agent_deliberation",
            "reason": trigger, "independent_corroboration": False,
            "required": ["consider available advice and disagreement", "record current evidence and rationale",
                         "label provisional analysis when consultation unavailable",
                         "obtain independently required human action approval"],
            "authorized": False, "automatic_paid_retry": False,
        }
        escalation = {"question":request.question,"state":packet["state"],"options":request.options,
                      "constraints":request.constraints,"primitive":request.primitive,"stakes":request.stakes,
                      "decision_type":request.decision_type,"unresolved":trigger,
                      "judgments":[{"provider":s.name,"decision":j.decision,"confidence":j.confidence,
                                    "confidence_kind":j.confidence_kind,"scores":j.scores} for s,j in judgments]}
        result["escalation_packet_bytes"] = len(encode(escalation))
        if len(encode(escalation)) > self.escalation_bytes:
            result["missing_information"] = ["Supply a smaller relevant state for deliberation; no evidence was silently truncated."]
            return finish(reason="ESCALATION_CONTEXT_REQUIRED")
        reasoning = choose({4})
        value = call(reasoning,escalation)
        if value is not None:
            result["escalated"] = True
            if not value.abstain and value.confidence >= threshold:
                return finish(value)
            return finish(reason="LOW_CONFIDENCE")
        # A caller can retrieve the bounded packet for its own configured reasoning provider.
        # It is returned, never stored in the evidence-minimized ledger.
        final = finish(reason=trigger if trigger in {"DECISION_UNSTABLE","LOW_CONFIDENCE"} else "ESCALATION_REQUIRED")
        return dict(final,escalation_packet=escalation)

    @staticmethod
    def deterministic(request, packet):
        rule = request.constraints.get("deterministic")
        if rule is None:
            return None
        if not isinstance(rule,dict):
            raise DecisionError("INVALID_RULE")
        if rule.get("op") == "threshold" and request.primitive == Primitive.BOOLEAN_PROBABILITY:
            value = packet["state"][rule["field"]]
            target = rule["value"]
            if not finite(value,-1e15,1e15) or not finite(target,-1e15,1e15):
                raise DecisionError("INVALID_RULE")
            comparisons = {"lt":value<target,"lte":value<=target,"gt":value>target,"gte":value>=target,"eq":value==target}
            decision = float(comparisons[rule["operator"]])
            return Judgment(decision,1.0,reason="DETERMINISTIC_RULE",confidence_kind="deterministic",provider_called=False,cost_usd=0)
        if rule.get("op") == "argmax" and request.primitive == Primitive.CHOICE:
            values = packet["state"][rule["field"]]
            if not isinstance(values,dict) or set(values)!=set(request.options) or not all(finite(v,-1e15,1e15) for v in values.values()):
                raise DecisionError("INVALID_RULE")
            ordered = sorted(values,key=lambda k:-values[k])
            if values[ordered[0]] == values[ordered[1]]:
                return Judgment(abstain=True,reason="DETERMINISTIC_TIE",provider_called=False,cost_usd=0)
            return Judgment(ordered[0],1.0,values,reason="DETERMINISTIC_RULE",confidence_kind="deterministic",provider_called=False,cost_usd=0)
        raise DecisionError("UNSUPPORTED_RULE")

    @staticmethod
    def stability(request, judgments):
        """Confidence/reliability weighted support with a strong-conflict veto; no majority vote."""
        def compatible(a,b):
            if request.primitive in {Primitive.CHOICE,Primitive.RANK}:
                return a.decision == b.decision
            scale = max(1,len(request.options)-1) if isinstance(request.options,list) else 1
            return abs(a.decision-b.decision)/scale <= .15
        def weight(spec,j):
            cal = spec.calibration.get(request.decision_type,{})
            reliability = cal.get("weight",1.0) if cal.get("samples",0)>=30 else 1.0
            if not finite(reliability,.1,1):
                reliability = 1.0
            return max(.001,j.confidence**4)*reliability
        winner_pair = max(judgments,key=lambda pair:sum(weight(s,j) for s,j in judgments if compatible(pair[1],j)))
        winner = Judgment(**asdict(winner_pair[1]))
        total = sum(weight(s,j) for s,j in judgments)
        agreement = sum(weight(s,j) for s,j in judgments if compatible(winner,j))/total
        strong_conflict = any(not compatible(a,b) and min(a.confidence,b.confidence)>=.65
                              for _,a in judgments for _,b in judgments)
        unstable = strong_conflict or agreement < .8
        winner.confidence = min(winner.confidence,agreement)
        return winner,round(agreement,6),unstable
