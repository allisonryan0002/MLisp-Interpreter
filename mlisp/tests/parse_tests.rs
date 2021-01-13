#[cfg(test)]
mod parse_tests {
    use mlisp::lex::Token;
    use mlisp::parse::parse;
    use mlisp::types::Expr;

    #[test]
    fn parse_fnum() {
        parse(&[Token::Literal("1".into())]).map_or_else(
            |err| assert!(false, format!("{:?}", err)),
            |expr| assert_eq!(Expr::fnum(1.0), expr),
        );
    }

    #[test]
    fn parse_symbol() {
        parse(&[Token::Literal("hello".into())]).map_or_else(
            |err| assert!(false, format!("{:?}", err)),
            |expr| assert_eq!(Expr::symbol("hello"), expr),
        )
    }

    #[test]
    fn parse_list() {
        let tokens = [
            Token::LPar,
            Token::Literal("+".into()),
            Token::Literal("1.8".into()),
            Token::Literal("1.2".into()),
            Token::RPar,
        ];
        let expected = Expr::list(&[Expr::symbol("+"), Expr::fnum(1.8), Expr::fnum(1.2)]);

        parse(&tokens).map_or_else(
            |err| assert!(false, format!("{:?}", err)),
            |expr| assert_eq!(expected, expr),
        );
    }

    #[test]
    fn parse_nested_symbol() {
        let tokens = [
            Token::LPar,
            Token::LPar,
            Token::Literal("A".into()),
            Token::RPar,
            Token::RPar,
        ];

        let expected = Expr::list(&[Expr::list(&[Expr::symbol("A")])]);

        parse(&tokens).map_or_else(
            |err| assert!(false, format!("{:?}", err)),
            |expr| assert_eq!(expected, expr),
        );
    }

    #[test]
    fn nested_lists() {
        let tokens = [
            Token::LPar,
            Token::Literal("+".into()),
            Token::LPar,
            Token::Literal("+".into()),
            Token::Literal("1.2".into()),
            Token::Literal("-12.8".into()),
            Token::RPar,
            Token::LPar,
            Token::Literal("*".into()),
            Token::Literal("1.2".into()),
            Token::Literal("-12.8".into()),
            Token::RPar,
            Token::RPar,
        ];

        let expected = Expr::list(&[
            Expr::symbol("+"),
            Expr::list(&[Expr::symbol("+"), Expr::fnum(1.2), Expr::fnum(-12.8)]),
            Expr::list(&[Expr::symbol("*"), Expr::fnum(1.2), Expr::fnum(-12.8)]),
        ]);

        parse(&tokens).map_or_else(
            |err| assert!(false, format!("{:?}", err)),
            |expr| assert_eq!(expected, expr),
        );
    }
}
