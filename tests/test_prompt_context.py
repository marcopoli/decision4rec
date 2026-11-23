from decision4rec.data.loaders import make_synthetic
from decision4rec.data.schema import Case
from decision4rec.prompts import PromptBuilder


def _case(ds, u, k):
    h = [int(x[0]) for x in ds.seqs[u][:k]]
    return Case(cid=0, user=u, history=h, target=int(ds.seqs[u][k][0]), cands=[1, 2], rscores=[0.0, 0.0], target_pos=1)


def test_context_marks_follow_the_shown_history_and_leave_the_default_unchanged():
    ds, _ = make_synthetic(n_users=40, n_items=60, seed=2)
    u = next(i for i, s in enumerate(ds.seqs) if len(s) >= 14)
    case = _case(ds, u, 12)
    plain = PromptBuilder(ds, None, hist_len=10).state(case)
    assert "before" not in plain and "same day" not in plain
    pb = PromptBuilder(ds, None, hist_len=10, context_marks="rating+days", day=1.0)
    marks = pb.marks(case)
    assert len(marks) == 10 and all(marks)
    s = ds.seqs[u]
    gap = int(float(s[11][1]) - float(s[2][1]))
    assert ("same day" if gap < 1 else f"{gap} day") in marks[0] and marks[-1].endswith("same day")
    assert pb.state(case).count("(") >= 10
    # a history that is not a prefix of the user's sequence gets no marks
    bad = Case(cid=0, user=u, history=list(reversed(case.history)), target=case.target, cands=[1, 2],
               rscores=[0.0, 0.0], target_pos=1)
    assert pb.marks(bad) == [] or case.history == list(reversed(case.history))
    # dropping an item removes its line and its mark together
    assert len(pb.marks(case, drop=case.history[-1])) == len([h for h in case.history[-10:] if h != case.history[-1]])
