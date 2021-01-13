use std::rc::Rc;


#[derive(Debug)]
pub enum Expr {
    Symbol(String),
    FNum(f64),
    List(Vec<Rc<Expr>>),
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Expr::Symbol(s1), Expr::Symbol(s2)) => s1 == s2,
            (Expr::FNum(n1), Expr::FNum(n2)) => (n1 - n2).abs() <= 1e-8,
            (Expr::List(xs1), Expr::List(xs2)) => xs1 == xs2,
            _ => false,
        }
    }
}

impl Eq for Expr {}

impl Expr {
    pub fn symbol(s: &str) -> Rc<Expr> {
        Rc::new(Expr::Symbol(s.to_string()))
    }

    pub fn fnum(n: f64) -> Rc<Expr> {
        Rc::new(Expr::FNum(n))
    }

    pub fn list(xs: &[Rc<Expr>]) -> Rc<Expr> {
        Rc::new(Expr::List(xs.iter().cloned().collect()))
    }

}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn build_symbol() {
        let sym_str = "a";
        let sym = Expr::symbol(&sym_str);
        match &*sym {
            Expr::Symbol(s) => assert_eq!(sym_str, s),
            _ => assert!(false),
        }
    }

    #[test]
    fn build_fnum() {
        let val = 1.0;
        let fnum = Expr::fnum(val);
        match &*fnum {
            Expr::FNum(n) => assert!((n - val).abs() <= 1e-8),
            _ => assert!(false),
        }
    }

    #[test]
    fn check_fnum_equality() {
        assert_eq!(Expr::FNum(1.0), Expr::FNum(1.0));
        assert_eq!(Expr::FNum(0.99), Expr::FNum(0.99));
    }

    #[test]
    fn build_list() {
        let vals = Expr::list(&[Expr::symbol("+"), Expr::fnum(1.0), Expr::fnum(1.0)]);
        match &*vals {
            Expr::List(xs) => {
                assert_eq!(xs.len(), 3);
                assert_eq!(Expr::symbol("+"), xs[0]);
                assert_eq!(Expr::fnum(1.0), xs[1]);
                assert_eq!(Expr::fnum(1.0), xs[2]);
            },
            _ => assert!(false),
        }
    }
}
