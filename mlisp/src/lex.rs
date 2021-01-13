#[derive(Debug)]
pub enum Token {
    LPar,
    RPar,
    Literal(String),
}

impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Token::Literal(l1), Token::Literal(l2)) => l1 == l2,
            (Token::LPar, Token::LPar)
            | (Token::RPar, Token::RPar) => true,
            _ => false,
        }
    }
}

#[derive(Debug)]
pub enum LexError {
    UnknownToken(String),
}

fn add_whitespace(input: &str) -> String {
    input
        .chars()
        .fold(Vec::new(), |mut acc, c| {
            if c == '(' || c == ')' {
                acc.append(&mut vec![' ', c, ' ']);
            } else {
                acc.push(c);
            }
            acc
        })
        .iter()
        .collect()
}

pub fn lex(input: &str) -> Result<Vec<Token>, LexError> {
    let r = add_whitespace(input)
        .split_ascii_whitespace()
        .map(|t| {
            match t {
                "(" => Token::LPar,
                ")" => Token::RPar,
                _ => Token::Literal(t.to_string()),
            }
        })
        .collect();
    Ok(r)
}

#[cfg(test)]
mod test {
    use super::*;
}
