#[cfg(test)]
mod interpreter_tests {
    use mlisp::eval::EvalResult;
    use mlisp::interpreter::run_interpreter;
    use mlisp::types::Expr;

    #[test]
    fn simple_statement() {
        let program = "(+ 1 (- 3 2))";
        let r = run_interpreter(&program);
        assert_eq!(EvalResult::Expr(Expr::fnum(2.0)), r);
    }

    #[test]
    fn complex_program() {
        let program = "((fn add-1 (x) (+ x 1))
        (let x 3)
        (let y (add-1 x))
        (let z (+ x y))
        (= z (+ x y)))";
        let r = run_interpreter(&program);
        assert_eq!(EvalResult::Expr(Expr::list(&[Expr::symbol("True")])), r);
    }

    #[test]
    fn bad_parse() {
        // Note: missing a ")" on the last line, which should case a parse error
        let program = "((fn add-1 (x) (+ x 1))
        (let x 3)
        (let y (add-1 x))
        (let z (+ x y))
        (= z (+ x y))";
        match run_interpreter(&program) {
            EvalResult::Err(_) => {}
            _ => assert!(
                false,
                "Expected EvalResult::Err resulting from a bad parse."
            ),
        }
    }
}
