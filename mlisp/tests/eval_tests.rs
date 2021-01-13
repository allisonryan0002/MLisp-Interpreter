#[cfg(test)]
mod eval_tests {
    use mlisp::eval::{eval, gen_print_output, Environment, EvalResult};
    use mlisp::types::Expr;

    #[test]
    fn add_and_check_simple_var_in_env() {
        // Check (let x 5) works
        let expected = 5.0;
        let vals = Expr::list(&[Expr::symbol("let"), Expr::symbol("x"), Expr::fnum(expected)]);
        let mut env = Environment::empty();
        env.push_context();

        // Step 1: evaluate expr
        let r1 = eval(vals, &mut env);
        assert_eq!(r1, EvalResult::Unit);

        // Step 2: check x = 5 is in environment
        let lookup = Expr::symbol("x");
        let r2 = eval(lookup.clone(), &mut env);
        assert_eq!(EvalResult::Expr(Expr::fnum(expected)), r2);

        // Step 3: pop context and check variable is gone
        env.pop_context();
        let r3 = eval(lookup.clone(), &mut env);
        assert_eq!(EvalResult::Expr(lookup.clone()), r3);
    }

    #[test]
    fn evaluate_symbol_in_env() {
        let val = Expr::fnum(1.0);
        let mut env = Environment::from_vars(&[("a", val.clone())]);

        let sym = Expr::symbol("a");
        let r = eval(sym.clone(), &mut env);
        assert_eq!(EvalResult::Expr(val.clone()), r);
    }

    #[test]
    fn evaluate_symbol_not_in_env() {
        let val = Expr::fnum(1.0);
        let mut env = Environment::from_vars(&[("a", val.clone())]);

        let s = "b";
        let sym = Expr::symbol(s);
        let r = eval(sym.clone(), &mut env);
        assert_eq!(EvalResult::Expr(sym), r);
    }

    #[test]
    fn add_and_check_expr_var_in_env() {
        // Check (let x 5) works
        let expected = 15.0;
        let vals = Expr::list(&[
            Expr::symbol("let"),
            Expr::symbol("x"),
            Expr::list(&[
                Expr::symbol("+"),
                Expr::fnum(1.0),
                Expr::fnum(2.0),
                Expr::fnum(3.0),
                Expr::fnum(4.0),
                Expr::fnum(5.0),
            ]),
        ]);
        let mut env = Environment::empty();
        env.push_context();

        // Step 1: evaluate expr
        let r1 = eval(vals, &mut env);
        assert_eq!(r1, EvalResult::Unit);

        // Step 2: check x = 5 is in environment
        let lookup = Expr::symbol("x");
        let r2 = eval(lookup.clone(), &mut env);
        assert_eq!(EvalResult::Expr(Expr::fnum(expected)), r2);

        // Step 3: pop context and check variable is gone
        env.pop_context();
        let r3 = eval(lookup.clone(), &mut env);
        assert_eq!(EvalResult::Expr(lookup.clone()), r3);
    }

    #[test]
    fn arithmetic_op_add_works() {
        let expr = Expr::list(&[
            Expr::symbol("+"),
            Expr::fnum(1.0),
            Expr::fnum(2.0),
            Expr::fnum(3.0),
            Expr::fnum(4.0),
            Expr::fnum(5.0),
        ]);
        let expected_sum = Expr::fnum(15.0);
        let mut env = Environment::empty();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected_sum), r);
    }

    #[test]
    fn arithmetic_op_sub_works() {
        let expr = Expr::list(&[
            Expr::symbol("-"),
            Expr::fnum(10.0),
            Expr::fnum(2.0),
            Expr::fnum(3.0),
            Expr::fnum(5.0),
        ]);
        let expected_sum = Expr::fnum(0.0);
        let mut env = Environment::empty();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected_sum), r);
    }

    #[test]
    fn arithmetic_op_mul_works() {
        let expr = Expr::list(&[
            Expr::symbol("*"),
            Expr::fnum(1.0),
            Expr::fnum(2.0),
            Expr::fnum(3.0),
            Expr::fnum(4.0),
        ]);
        let expected_sum = Expr::fnum(24.0);
        let mut env = Environment::empty();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected_sum), r);
    }

    #[test]
    fn arithmetic_op_div_works() {
        let expr = Expr::list(&[
            Expr::symbol("/"),
            Expr::fnum(24.0),
            Expr::fnum(2.0),
            Expr::fnum(3.0),
            Expr::fnum(4.0),
        ]);
        let expected = Expr::fnum(1.0);
        let mut env = Environment::empty();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected), r);
    }

    #[test]
    fn equality_works_1() {
        let expr = Expr::list(&[
            Expr::symbol("="),
            Expr::fnum(1.0),
            Expr::fnum(1.0),
            Expr::fnum(1.0),
            Expr::fnum(1.0),
        ]);
        let expected = Expr::symbol("True");
        let mut env = Environment::empty();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected), r);
    }

    #[test]
    fn equality_works_2() {
        let expr = Expr::list(&[
            Expr::symbol("="),
            Expr::symbol("a"),
            Expr::symbol("b"),
            Expr::symbol("a"),
            Expr::symbol("a"),
        ]);
        let expected = Expr::symbol("False");
        let mut env = Environment::empty();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected), r);
    }

    #[test]
    fn equality_works_3() {
        let expr = Expr::list(&[
            Expr::symbol("="),
            Expr::list(&[
                Expr::symbol("a"),
                Expr::symbol("b"),
            ]),
            Expr::list(&[
                Expr::symbol("a"),
                Expr::symbol("a"),
            ]),
        ]);
        let expected = Expr::symbol("False");
        let mut env = Environment::empty();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected), r);
    }

    #[test]
    fn equality_works_4() {
        let expr = Expr::list(&[
            Expr::symbol("="),
            Expr::list(&[
                Expr::symbol("a"),
                Expr::symbol("a"),
            ]),
            Expr::list(&[
                Expr::symbol("a"),
                Expr::symbol("a"),
            ]),
        ]);
        let expected = Expr::symbol("True");
        let mut env = Environment::empty();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected), r);
    }

    #[test]
    fn inequality_works_1() {
        let expr = Expr::list(&[
            Expr::symbol("!="),
            Expr::fnum(1.0),
            Expr::fnum(1.0),
            Expr::fnum(1.0),
            Expr::fnum(1.0),
        ]);
        let expected = Expr::symbol("False");
        let mut env = Environment::empty();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected), r);
    }

    #[test]
    fn inequality_works_2() {
        let expr = Expr::list(&[
            Expr::symbol("!="),
            Expr::symbol("a"),
            Expr::symbol("b"),
            Expr::symbol("a"),
            Expr::symbol("a"),
        ]);
        let expected = Expr::symbol("True");
        let mut env = Environment::empty();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected), r);
    }

    #[test]
    fn nequality_works_3() {
        let expr = Expr::list(&[
            Expr::symbol("!="),
            Expr::list(&[
                Expr::symbol("a"),
                Expr::symbol("b"),
            ]),
            Expr::list(&[
                Expr::symbol("a"),
                Expr::symbol("a"),
            ]),
        ]);
        let expected = Expr::symbol("True");
        let mut env = Environment::empty();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected), r);
    }

    #[test]
    fn inequality_works_4() {
        let expr = Expr::list(&[
            Expr::symbol("!="),
            Expr::list(&[
                Expr::symbol("a"),
                Expr::symbol("a"),
            ]),
            Expr::list(&[
                Expr::symbol("a"),
                Expr::symbol("a"),
            ]),
        ]);
        let expected = Expr::symbol("False");
        let mut env = Environment::empty();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected), r);
    }

    #[test]
    fn boolean_op_or_works_1() {
        let expr = Expr::list(&[
            Expr::symbol("or"),
            Expr::symbol("True"),
            Expr::symbol("False"),
            Expr::symbol("False"),
            Expr::symbol("False"),
        ]);
        let expected = Expr::symbol("True");
        let mut env = Environment::default();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected), r);
    }

    #[test]
    fn boolean_op_or_works_2() {
        let expr = Expr::list(&[
            Expr::symbol("or"),
            Expr::symbol("True"),
            Expr::symbol("True"),
            Expr::symbol("True"),
        ]);
        let expected = Expr::symbol("True");
        let mut env = Environment::default();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected), r);
    }

    #[test]
    fn boolean_op_and_works_1() {
        let expr = Expr::list(&[
            Expr::symbol("and"),
            Expr::symbol("True"),
            Expr::symbol("False"),
            Expr::symbol("True"),
            Expr::symbol("False"),
        ]);
        let expected = Expr::symbol("False");
        let mut env = Environment::default();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected), r);
    }

    #[test]
    fn boolean_op_and_works_2() {
        let expr = Expr::list(&[
            Expr::symbol("and"),
            Expr::symbol("True"),
            Expr::symbol("True"),
            Expr::symbol("True"),
            Expr::symbol("True"),
        ]);
        let expected = Expr::symbol("True");
        let mut env = Environment::default();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected), r);
    }

    #[test]
    fn boolean_not_works_1() {
        let expr = Expr::list(&[
            Expr::symbol("not"),
            Expr::symbol("True"),
        ]);
        let expected = Expr::symbol("False");
        let mut env = Environment::default();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected), r);
    }

    #[test]
    fn boolean_not_works_2() {
        let expr = Expr::list(&[
            Expr::symbol("not"),
            Expr::symbol("False"),
        ]);
        let expected = Expr::symbol("True");
        let mut env = Environment::default();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected), r);
    }

    #[test]
    fn boolean_not_works_3() {
        let expr = Expr::list(&[
            Expr::symbol("not"),
            Expr::list(&[
                Expr::symbol("="),
                Expr::symbol("a"),
                Expr::symbol("a"),
            ]),
        ]);
        let expected = Expr::symbol("False");
        let mut env = Environment::default();
        let r = eval(expr, &mut env);
        assert_eq!(EvalResult::Expr(expected), r);
    }
    #[test]
    fn add_fn_to_env() {
        let x1_sym = "x1";
        let x2_sym = "x2";
        let x1 = Expr::symbol(&x1_sym);
        let x2 = Expr::symbol(&x2_sym);
        let params = Expr::list(&[x1.clone(), x2.clone()]);
        let fn_body = Expr::list(&[Expr::symbol("+"), x1.clone(), x2.clone()]);
        let f_name = "test-func";
        let expr = Expr::list(&[
            Expr::symbol("fn"),
            Expr::symbol(&f_name),
            params.clone(),
            fn_body.clone(),
        ]);
        let mut env = Environment::empty();
        env.push_context();

        let r = eval(expr, &mut env);
        assert_eq!(r, EvalResult::Unit);

        env.lookup(&f_name).map_or_else(
            || assert!(false, "Expected function in environment but got None"),
            |(params, body)| {
                assert_eq!(&params[0], x1_sym);
                assert_eq!(&params[1], x2_sym);
                assert_eq!(body, fn_body);
            },
        );
    }

    #[test]
    fn test_func_evaluation() {
        let x = 2.0;
        let x_name = Expr::symbol("x");
        let y = 3.0;
        let y_name = Expr::symbol("y");

        let fn_name = Expr::symbol("test-func");

        // (fn test-func (x y) (+ x y))
        let fn_def = Expr::list(&[
            Expr::symbol("fn"),
            fn_name.clone(),
            Expr::list(&[x_name.clone(), y_name.clone()]),
            Expr::list(&[Expr::symbol("+"), x_name.clone(), y_name.clone()]),
        ]);

        let mut env = Environment::empty();
        env.push_context();
        let r = eval(fn_def, &mut env);
        assert_eq!(EvalResult::Unit, r);

        // (test-func (+ 2 3) 1) -> Expect Expr:FNum(6.0)
        let fn_eval = Expr::list(&[
            fn_name.clone(),
            Expr::list(&[Expr::symbol("+"), Expr::fnum(x), Expr::fnum(y)]),
            Expr::fnum(1.0),
        ]);
        let r2 = eval(fn_eval, &mut env);
        if let EvalResult::Expr(e) = r2 {
            if let Expr::FNum(n) = *e {
                assert_eq!(n, x + y + 1.0);
            } else {
                assert!(false, format!("Expected FNum(6.0), got {:?}", e));
            }
        } else {
            assert!(false, format!("Expected Expr::fnum(6.0), got {:?}", r2));
        }
    }

    #[test]
    fn test_eval_ite() {
        let e = Expr::list(&[
            Expr::symbol("if"),
            Expr::list(&[Expr::symbol("True")]),
            Expr::list(&[Expr::symbol("x")]),
            Expr::list(&[Expr::symbol("y")]),
        ]);

        let mut env = Environment::default();
        let result = eval(e, &mut env);
        if let EvalResult::Expr(expr) = result {
            assert_eq!(Expr::list(&[Expr::symbol("x".into())]), expr);
        } else {
            assert!(false, "Expected expression, got {:?}", result);
        }
    }

    #[test]
    fn test_ite_true_block() {
        let e = Expr::list(&[
            Expr::symbol("if"),
            Expr::list(&[Expr::symbol("True")]),
            Expr::list(&[Expr::symbol("x")]),
            Expr::list(&[Expr::symbol("y")]),
        ]);
        let mut env = Environment::default();
        let result = eval(e.clone(), &mut env);
        if let EvalResult::Expr(expr) = result {
            assert_eq!(Expr::list(&[Expr::symbol("x".into())]), expr);
        } else {
            assert!(false, "Expected expression, got {:?}", result);
        }
    }

    #[test]
    fn test_ite_false_block() {
        let e = Expr::list(&[
            Expr::symbol("if"),
            Expr::list(&[Expr::symbol("False")]),
            Expr::list(&[Expr::symbol("x")]),
            Expr::list(&[Expr::symbol("y")]),
        ]);

        let mut env = Environment::default();
        let result = eval(e.clone(), &mut env);
        if let EvalResult::Expr(expr) = result {
            assert_eq!(Expr::list(&[Expr::symbol("y".into())]), expr);
        } else {
            assert!(false, "Expected expression, got {:?}", result);
        }
    }

    #[test]
    fn test_print() {
        let e1 = Expr::symbol("hello");
        let e2 = Expr::fnum(3.2);
        let e3 = Expr::list(&[Expr::symbol("hello"), Expr::symbol("world")]);

        let mut env = Environment::empty();

        assert_eq!("hello", gen_print_output(e1.clone(), &mut env));
        assert_eq!("3.2", gen_print_output(e2.clone(), &mut env));
        assert_eq!("(hello world)", gen_print_output(e3.clone(), &mut env));

        env.push_context();
        env.add_fn(
            "test-func",
            &["x1".into(), "x2".into()],
            Expr::symbol("body"),
        )
        .map_or_else(
            |e| assert!(false, format!("got error {}", e)),
            |_| {
                assert_eq!(
                    "<func-object: test-func>",
                    gen_print_output(Expr::symbol("test-func"), &mut env)
                )
            },
        );
        let _ = env
            .add_var("x", Expr::fnum(42.0))
            .map_err(|e| assert!(false, format!("got error {}", e)));

        let e4 = Expr::list(&[Expr::symbol("test-func"), Expr::symbol("x"), e3.clone()]);
        assert_eq!(
            "(<func-object: test-func> 42 (hello world))",
            gen_print_output(e4.clone(), &mut env)
        );

        let e5 = Expr::list(&[
            Expr::symbol("print"),
            e4.clone(),
            e3.clone(),
            e2.clone(),
            e1.clone(),
        ]);
        eval(e5.clone(), &mut env);

        // In code: (print Hello world!)
        let e6 = Expr::list(&[
            Expr::symbol("print"),
            Expr::symbol("Hello"),
            Expr::symbol("world!"),
        ]);
        eval(e6.clone(), &mut env);
    }

}

#[cfg(test)]
mod environment_tests {
    use mlisp::eval::Environment;
    use mlisp::types::Expr;
    use std::collections::HashMap;
    use std::rc::Rc;

    #[test]
    fn cannot_add_to_contextless_env() {
        let mut env = Environment::empty();

        let r = env.add_var("a", Expr::fnum(1.0));
        assert!(
            r.is_err(),
            format!("Expected add_var to fail, but it succeeded: {:?}", r)
        );
    }

    #[test]
    fn push_context_works() {
        let mut env = Environment::empty();
        assert_eq!(0, env.contexts.len());
        env.push_context();
        assert_eq!(1, env.contexts.len());
    }

    #[test]
    fn pop_context_works() {
        let mut env = Environment::empty();
        env.push_context();
        assert_eq!(1, env.contexts.len());
        env.pop_context();
        assert_eq!(0, env.contexts.len());
    }

    #[test]
    fn default_environment_is_correct() {
        let env = Environment::default();
        env.lookup("False").map_or_else(
            || assert!(false, "Expected Some, got None"),
            |(ps, expr)| {
                assert_eq!(0, ps.len());
                assert_eq!(Expr::list(&[]), expr);
            },
        );
        env.lookup("True").map_or_else(
            || assert!(false, "Expected Some, got None"),
            |(ps, expr)| {
                assert_eq!(0, ps.len());
                assert_eq!(Expr::list(&[Expr::fnum(1.0)]), expr);
            },
        );
    }

    #[test]
    fn lookup_works() {
        let ctx = [("x".into(), (Vec::new(), Expr::fnum(1.0)))]
            .iter()
            .cloned()
            .collect::<HashMap<String, (Vec<String>, Rc<Expr>)>>();
        let env = Environment {
            contexts: vec![ctx],
        };
        env.lookup("x").map_or_else(
            || assert!(false, "Expected Some, got None"),
            |(ps, expr)| {
                assert_eq!(0, ps.len());
                assert_eq!(Expr::fnum(1.0), expr);
            },
        );
        assert!(env.lookup("y").is_none(), "Expected None, got Some");
    }

    #[test]
    fn contain_key_works() {
        let ctx = [("x".into(), (Vec::new(), Expr::fnum(1.0)))]
            .iter()
            .cloned()
            .collect::<HashMap<String, (Vec<String>, Rc<Expr>)>>();
        let env = Environment {
            contexts: vec![ctx],
        };
        assert!(env.contains_key("x"), "Environment should contain x.");
        assert!(
            !env.contains_key("y"),
            "Environment shoulnt not contain y but apparently does."
        );
    }

    #[test]
    fn add_var_to_context_works() {
        let mut env = Environment::empty();
        let val = Expr::fnum(1.0);
        assert_eq!(0usize, env.num_contexts());
        env.push_context();
        assert_eq!(1usize, env.num_contexts());

        // Insert variable
        let r = env.add_var("a", val.clone());
        assert!(r.is_ok());

        // Lookup the variable and validate
        env.lookup("a").map_or_else(
            || assert!(false, "Failed to find var in environment."),
            |(args, x)| {
                assert_eq!(val, x);
                assert_eq!(0usize, args.len());
            },
        );

        // Pop context and check variable is gone
        env.pop_context();
        env.lookup("a")
            .map(|x| assert!(false, format!("Expected Err, got {:?}", x)));
        assert_eq!(0usize, env.num_contexts());
    }

    #[test]
    fn add_fn_to_context_works() {
        let x1_sym = "x1";
        let x2_sym = "x2";
        let x1 = Expr::symbol(&x1_sym);
        let x2 = Expr::symbol(&x2_sym);
        let fn_body = Expr::list(&[Expr::symbol("+"), x1.clone(), x2.clone()]);
        let f_name = "test-func";

        let mut env = Environment::empty();
        env.push_context();
        assert_eq!(1, env.contexts.len());
        let _ = env.add_fn(
            &f_name,
            &[x1_sym.to_string(), x2_sym.to_string()],
            fn_body.clone(),
        );

        let _ = env.lookup(&f_name).map_or_else(
            || assert!(false, "Expected function in environment but got None"),
            |(params, body)| {
                assert_eq!(&params[0], x1_sym);
                assert_eq!(&params[1], x2_sym);
                assert_eq!(body, fn_body);
            },
        );
    }
}
