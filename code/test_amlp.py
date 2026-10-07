import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amlp import Envelope, control_values, lift


def env():
    return Envelope("banking", {"send_money"}, {("send_money", "recipient"): {"gb29nwbk60161331926819"}},
                    {("send_money", "recipient"): {"gb29nwbk60161331926819"}}, set(),
                    env_text='{"iban": "de89370400440532013000"}', prompt="Pay my rent to CH9300762011623852957.")


def test_read_tools_never_checked():
    assert env().violations("get_balance", {}) == []


def test_tool_outside_layer_blocked():
    assert env().violations("update_password", {"password": "x"}) == ["<tool>"]


def test_value_sources():
    e = env()
    assert e.violations("send_money", {"recipient": "GB29NWBK60161331926819", "amount": 1}) == []   # mined
    assert e.violations("send_money", {"recipient": "DE89370400440532013000"}) == []                # clean env
    assert e.violations("send_money", {"recipient": "CH9300762011623852957"}) == []                 # prompt
    assert e.violations("send_money", {"recipient": "US133000000121212121212"}) == ["recipient"]     # attacker


def test_empty_and_list_values():
    assert control_values("send_email", {"cc": None, "bcc": "[]", "recipients": "['a@x.com', 'b@y.com']"}) == \
        [("recipients", "a@x.com"), ("recipients", "b@y.com")]


def test_lift_only_on_lift_args():
    e = Envelope("workspace", {"send_email"}, {}, {("send_email", "recipients"): {"@corp.com"}},
                 {("send_email", "recipients")}, "", "")
    assert e.violations("send_email", {"recipients": ["new@corp.com"]}) == []
    assert e.violations("send_email", {"recipients": ["evil@gmail.com"]}) == ["recipients"]
    assert lift("a@b.com") == "@b.com" and lift("https-host") == "https-host"


def test_exclude_tasks_keeps_holdout_out_of_mining_pool():
    import json, os
    import envbuild as B
    h = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "holdout_split.json")))["holdout"]
    hold = frozenset((s, ut) for s, uts in h.items() for ut in uts)
    s, ut = sorted(k for k in B.BEN if k not in hold and k[0] == "banking")[0]
    full = B.build(s, ut, predictor=None, r=3, use_env=False)
    cut = B.build(s, ut, predictor=None, r=3, use_env=False, exclude_tasks=hold)
    mined_from_hold = {v for k in hold if k[0] == s and k != (s, ut) for o in B.BEN[k] for c in o["trace"]
                       for _, v in control_values(c["f"], c["a"])}
    only_hold = mined_from_hold - {v for k in B.BEN if k[0] == s and k not in hold and k != (s, ut)
                                   for o in B.BEN[k] for c in o["trace"] for _, v in control_values(c["f"], c["a"])}
    cut_vals = set().union(*cut.values.values()) if cut.values else set()
    assert not (only_hold & cut_vals)
    assert set().union(*full.values.values()) >= cut_vals
