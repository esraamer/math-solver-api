from fastapi import FastAPI
from pydantic import BaseModel
from sympy import symbols, Eq, S, simplify, factor, expand, diff, integrate
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers.solveset import solveset

app = FastAPI()

class Query(BaseModel):
    task: str
    expression: str
    variable: str = "x"
    showSteps: bool = True
    precision: int | None = None

@app.get("/health")
def health(): return {"ok": True}

@app.post("/math")
def math(q: Query):
    x = symbols(q.variable)
    expr_text = q.expression.replace("^", "**").strip()

    def out(payload):
        if q.precision is not None and "result" in payload:
            try:
                payload["result"] = str(simplify(payload["result"]).evalf(q.precision))
            except: pass
        return {"ok": True, **payload}

    try:
        if q.task == "solve":
            if "=" in expr_text:
                L, R = expr_text.split("=")
                eq = Eq(parse_expr(L), parse_expr(R))
            else:
                eq = Eq(parse_expr(expr_text), 0)
            sols = list(solveset(eq, x, domain=S.Complexes))
            verify = []
            for s in sols:
                verify.append(str(simplify(eq.lhs.subs(x, s) - eq.rhs.subs(x, s))) == "0")
            steps = [f"Solve {eq} for {q.variable}"] if q.showSteps else []
            return out({"type":"solve","solution":[str(s) for s in sols],
                        "verified": all(verify), "steps": steps})

        elif q.task == "simplify":
            expr = parse_expr(expr_text); res = simplify(expr)
            steps = [f"Simplify {expr} → {res}"] if q.showSteps else []
            return out({"type":"simplify","result":str(res), "steps": steps})

        elif q.task == "factor":
            expr = parse_expr(expr_text); res = factor(expr)
            steps = [f"Factor {expr} → {res}"] if q.showSteps else []
            return out({"type":"factor","result":str(res), "steps": steps})

        elif q.task == "expand":
            expr = parse_expr(expr_text); res = expand(expr)
            steps = [f"Expand {expr} → {res}"] if q.showSteps else []
            return out({"type":"expand","result":str(res), "steps": steps})

        elif q.task == "differentiate":
            expr = parse_expr(expr_text); res = diff(expr, x)
            steps = [f"d/d{q.variable}({expr}) = {res}"] if q.showSteps else []
            return out({"type":"differentiate","result":str(res), "steps": steps})

        elif q.task == "integrate":
            expr = parse_expr(expr_text); res = integrate(expr, x)
            steps = [f"∫ {expr} d{q.variable} = {res} + C"] if q.showSteps else []
            return out({"type":"integrate","result":str(res) + " + C", "steps": steps})

        else:
            return {"ok": False, "error": "Unknown task"}
    except Exception as e:
        return {"ok": False, "error": f"Parse/compute error: {e}"}
