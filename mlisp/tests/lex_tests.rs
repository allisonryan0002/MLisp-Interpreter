#[cfg(test)]
mod lex_tests {
    use mlisp::lex::{lex, Token};

    #[test]
    fn can_lex_lpar() {
        lex("(").map_or_else(
            |err| assert!(false, format!("{:?}", err)),
            |tokens| {
                assert_eq!(1, tokens.len());
                assert_eq!(Token::LPar, tokens[0]);
            },
        );
    }

    #[test]
    fn can_lex_rpar() {
        lex(")").map_or_else(
            |err| assert!(false, format!("{:?}", err)),
            |tokens| {
                assert_eq!(1, tokens.len());
                assert_eq!(Token::RPar, tokens[0]);
            },
        );
    }

    #[test]
    fn can_lex_literal() {
        lex("hello-world").map_or_else(
            |err| assert!(false, format!("{:?}", err)),
            |tokens| {
                assert_eq!(1, tokens.len());
                assert_eq!(Token::Literal("hello-world".into()), tokens[0]);
            },
        );
    }

    #[test]
    fn lex_test_1() {
        let input = "(+ 10 -2)";
        let output = vec![
            Token::LPar,
            Token::Literal("+".into()),
            Token::Literal("10".into()),
            Token::Literal("-2".into()),
            Token::RPar,
        ];

        match lex(input) {
            Ok(actual) => assert_eq!(output, actual),
            _ => assert!(false),
        }
    }

    #[test]
    fn lex_test_2() {
        let input = "(let x (some-fn 1 2 3))";
        let output = vec![
            Token::LPar,
            Token::Literal("let".into()),
            Token::Literal("x".into()),
            Token::LPar,
            Token::Literal("some-fn".into()),
            Token::Literal("1".into()),
            Token::Literal("2".into()),
            Token::Literal("3".into()),
            Token::RPar,
            Token::RPar,
        ];

        match lex(input) {
            Ok(actual) => assert_eq!(output, actual),
            _ => assert!(false),
        }
    }

    #[test]
    fn lex_empty_string() {
        lex("").map_or_else(
            |err| assert!(false, format!("{:?}", err)),
            |tokens| assert_eq!(0, tokens.len()),
        );
    }
}
