use std::rc::Rc;

#[derive(Debug, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

pub enum Expr {
    Binary(Op, Rc<Expr>, Rc<Expr>),
    Numeric(f64),
}

impl Expr {
    pub fn binary(op: Op, lhs: Rc<Expr>, rhs: Rc<Expr>) -> Rc<Expr> {
        Rc::new(Expr::Binary(op, lhs, rhs))
    }

    pub fn numeric(n: f64) -> Rc<Expr> {
        Rc::new(Expr::Numeric(n))
    }

    pub fn eval(&self) -> f64 {
        match self {
            Expr::Numeric(n) => *n,
            Expr::Binary(op, lhs, rhs) => match op {
                Op::Add => lhs.eval() + rhs.eval(),
                Op::Sub => lhs.eval() - rhs.eval(),
                Op::Mul => lhs.eval() * rhs.eval(),
                Op::Div => lhs.eval() / rhs.eval(),
            },
        }
    }
}

pub struct ExprTree {
    root: Option<Rc<Expr>>,
}

impl ExprTree {
    pub fn new() -> ExprTree {
        ExprTree { root: None }
    }

    pub fn from_expr(expr: Rc<Expr>) -> ExprTree {
        ExprTree { root: Some(expr) }
    }

    pub fn eval(&self) -> Option<f64> {
        match &self.root {
            None => None,
            Some(expr) => Some(expr.eval()),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_none_eval() {
        let et = ExprTree::new();
        assert_eq!(None, et.eval());
    }

    #[test]
    fn test_some_eval() {
        let et = ExprTree::from_expr(Expr::binary(
            Op::Add,
            Expr::numeric(1.0),
            Expr::numeric(2.0),
        ));
        assert_eq!(Some(3.0), et.eval());
    }

    #[test]
    fn test_numeric_expr() {
        let e = Expr::numeric(3.0);
        assert_eq!(3.0, e.eval())
    }

    #[test]
    fn test_binary_expr_add() {
        let e = Expr::binary(Op::Add, Expr::numeric(1.0), Expr::numeric(2.0));
        assert_eq!(3.0, e.eval());
    }

    #[test]
    fn test_binary_expr_sub() {
        let e = Expr::binary(Op::Sub, Expr::numeric(1.0), Expr::numeric(2.0));
        assert_eq!(-1.0, e.eval());
    }

    #[test]
    fn test_binary_expr_mul() {
        let e = Expr::binary(Op::Mul, Expr::numeric(1.0), Expr::numeric(2.0));
        assert_eq!(2.0, e.eval());
    }

    #[test]
    fn test_binary_expr_div() {
        let e = Expr::binary(Op::Div, Expr::numeric(1.0), Expr::numeric(2.0));
        assert_eq!(0.5, e.eval());
    }
}
