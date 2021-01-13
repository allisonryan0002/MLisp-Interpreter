use crate::types::Expr;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, PartialEq)]
pub enum EvalResult {
    Err(String),
    Expr(Rc<Expr>),
    Unit,
}

#[derive(Debug)]
pub struct Environment {
    pub contexts: Vec<HashMap<String, (Vec<String>, Rc<Expr>)>>,
}

impl Environment {
    pub fn empty() -> Environment {
        Environment {
            contexts: Vec::new(),
        }
    }

    /// Helper function for tests
    pub fn from_vars(vars: &[(&str, Rc<Expr>)]) -> Environment {
        let mut env = Environment::empty();
        env.push_context();
        vars.iter().for_each(|(name, expr)| {
            let _ = env.add_var(name, expr.clone());
        });
        env
    }

    pub fn default() -> Environment {
        let ctx = [("False".into(), (Vec::new(), Expr::list(&Vec::new()))),
            ("True".into(), (Vec::new(), Expr::list(&vec![Expr::fnum(1.0)])))]
            .iter()
            .cloned()
            .collect::<HashMap<String, (Vec<String>, Rc<Expr>)>>();
        let env = Environment {
            contexts: vec![ctx],
        };
        env
    }

    /// Looks up the given symbol in the Environment.
    pub fn lookup(&self, symbol: &str) -> Option<(Vec<String>, Rc<Expr>)> {
        self.contexts
            .iter()
            .find(|ctx| ctx.contains_key(symbol))
            .map(|ctx| ctx.get(symbol))
            .flatten()
            .cloned()
    }

    /// Checks whether the given symbol exists in the Environment.
    pub fn contains_key(&self, symbol: &str) -> bool {
        self.contexts
            .iter()
            .rev()
            .find(|ctx| ctx.contains_key(symbol))
            .is_some()
    }

    /// Pushes a new context on the `contexts` stack.
    pub fn push_context(&mut self) {
        self.contexts.push(HashMap::new());
    }

    /// Pops the last context from the `contexts` stack.
    pub fn pop_context(&mut self) {
        self.contexts.pop();
    }

    /// Adds a variable definition to the Environment
    pub fn add_var(&mut self, var: &str, val: Rc<Expr>) -> Result<(), String> {
        self.contexts
            .last_mut()
            .map_or_else(
                || Err("No context to add variable to".into()),
                |ctx| { ctx.insert(var.to_string(), (Vec::new(), val.clone())); Ok(()) },
            )
    }

    /// Adds a function definition to the Environment
    pub fn add_fn(&mut self, name: &str, params: &[String], body: Rc<Expr>) -> Result<(), String> {
        self.contexts
            .last_mut()
            .map_or_else(
                || Err("No context to add variable to".into()),
                |ctx| { ctx.insert(name.to_string(), (params.to_vec(), body.clone())); Ok(()) },
            )
    }

    pub fn num_contexts(&self) -> usize {
        self.contexts.len()
    }
}

/// Generates the output printed to standard out when the user calls print.
pub fn gen_print_output(expr: Rc<Expr>, env: &mut Environment) -> String {
    match &*expr {
        Expr::Symbol(s) => {
            match env.lookup(&s) {
                None => s.to_string(),
                Some((params, e)) if params.len() == 0 => gen_print_output(e, env),
                _ => format!("<func-object: {}>", s.to_string()),
            }
        },
        Expr::FNum(n) => format!("{}", n),
        Expr::List(vals) => {
            let vals_out: Vec<String> = vals.iter()
                .cloned()
                .map(|x| gen_print_output(x, env))
                .collect();
            format!("({})", vals_out.join(" "))
        }
    }
}

fn evaluate_symbol(expr: Rc<Expr>, sym: &str, args: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    env.lookup(sym).map_or_else(
        || EvalResult::Expr(expr),
        |(param_names, expression)| {
            if param_names.is_empty() {
                eval(expression.clone(), env)
            } else {
                if args.len() != param_names.len() {
                    return EvalResult::Err("Mismatched input params".into());
                }
                let mapped_args: Result<Vec<(String, Rc<Expr>)>, String> = args.iter()
                    .zip(param_names)
                    .map(|(expr, name)| match eval(expr.clone(), env) {
                        EvalResult::Expr(e) => Ok((name.to_string(), e.clone())),
                        EvalResult::Err(err) => Err(err),
                        _ => Err("Cannot pass Unit as function arg".into()),
                    })
                    .collect();
                env.push_context();
                let result = mapped_args.map_or_else(
                    |err| EvalResult::Err(err),
                    |arg_tuples| {
                        arg_tuples.iter().for_each(|(name, expr)| { let _ = env.add_var(name, expr.clone()); });
                        eval(expression.clone(), env)
                    }
                );
                env.pop_context();
                result
            }
        }
    )
}

fn add_var_to_env(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.len() != 2 {
        return EvalResult::Err("Invalid varible definition".into());
    }
    let restricted_strings = vec!["+", "-", "*", "/", "or", "and", "not", "=", "!=", "if", "let", "fn", "print"];
    match (&*vals[0], &vals[1]) {
        (Expr::Symbol(s), e) => {
            if restricted_strings.iter().any(|&x| x == s) {
                return EvalResult::Err("Cannot define a restricted key word".into());
            }
            match eval(e.clone(), env) {
                EvalResult::Expr(e) => {
                    env.add_var(s, e)
                        .map_or_else(
                            |s| EvalResult::Err(s),
                            |_| EvalResult::Unit,
                        )
                },
                EvalResult::Unit => EvalResult::Err("Cannot asign Unit to variable".into()),
                err => err,
            }
        },
        _ => EvalResult::Err("Must be symbol followed by an expression".into()),
    }
}

fn add_fn_to_env(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.len() != 3 {
        EvalResult::Err("Function definition must have exactly 3 elements".into());
    }
    let restricted_strings = vec!["+", "-", "*", "/", "or", "and", "not", "=", "!=", "if", "let", "fn", "print"];
    let fn_name = &*vals[0];
    let p_names = &*vals[1];
    let body = &vals[2];
    match(&*fn_name, p_names, body) {
        (Expr::Symbol(name), Expr::List(params), body) => {
            if restricted_strings.iter().any(|&x| x == name) {
                return EvalResult::Err("Cannot define a restricted key word".into());
            }
            let ps: Result<Vec<String>, String> = params.iter().cloned().map(|e| {
                if let Expr::Symbol(n) = &*e {
                    Ok(n.to_string())
                } else {
                    Err("Function params need to be symbols".into())
                }
            })
            .collect();

            ps.map_or_else(
                |err| EvalResult::Err(err),
                |xs| env.add_fn(name, &xs, body.clone()).map_or_else(
                    |err| EvalResult::Err(err),
                    |_| EvalResult::Unit,
                )
            )
        },
        _ => EvalResult::Err("Function definition has incorrect types".into())
    }
}

fn add_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.is_empty() {
        return EvalResult::Err("Need at least one number".into());
    }
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env) {
            EvalResult::Expr(expr) => match &*expr {
                Expr::FNum(n) => Ok(*n),
                _ => Err("Can only sum numbers".into()),
            },
            _ => Err("Can only sum numbers".into()),
        })
        .collect::<Result<Vec<f64>, String>>();
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| EvalResult::Expr(Expr::fnum(xs.iter().sum())),
    )
}

fn sub_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.is_empty() {
        return EvalResult::Err("Need at least one number".into());
    }
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env) {
            EvalResult::Expr(expr) => match &*expr {
                Expr::FNum(n) => Ok(*n),
                _ => Err("Can only subtract numbers".into()),
            },
            _ => Err("Can only subract numbers".into()),
        })
        .collect::<Result<Vec<f64>, String>>();
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| {
            let first = xs[0]; 
            EvalResult::Expr(Expr::fnum(xs[1..].iter().fold(first, |acc, x| acc - x))) 
        },
    )
}

fn mul_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.is_empty() {
        return EvalResult::Err("Need at least one number".into());
    }
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env) {
            EvalResult::Expr(expr) => match &*expr {
                Expr::FNum(n) => Ok(*n),
                _ => Err("Can only multiply numbers".into()),
            },
            _ => Err("Can only multiply numbers".into()),
        })
        .collect::<Result<Vec<f64>, String>>();
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| EvalResult::Expr(Expr::fnum(xs.iter().fold(1.0, |acc, x| acc * x)))
    )
}

fn div_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.is_empty() {
        return EvalResult::Err("Need at least one number".into());
    }
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env) {
            EvalResult::Expr(expr) => match &*expr {
                Expr::FNum(n) => Ok(*n),
                _ => Err("Can only divide numbers".into()),
            },
            _ => Err("Can only divide numbers".into()),
        })
        .collect::<Result<Vec<f64>, String>>();
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| {
            let first = xs[0]; 
            EvalResult::Expr(Expr::fnum(xs[1..].iter().fold(first, |acc, x| acc / x))) 
        },
    )
}

fn eq_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.is_empty() {
        return EvalResult::Err("Need at least one argument".into());
    }
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env) {
            EvalResult::Expr(expr) => Ok(expr),
            _ => Err("Can only test equality on numbers and symbols".into())
        })
        .collect::<Result<Vec<Rc<Expr>>, String>>();
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| {
            for i in 1..xs.len() {
                if xs[i] != xs[i-1] {
                    return EvalResult::Expr(Expr::symbol("False"));
                }
            }
            EvalResult::Expr(Expr::symbol("True"))
        }
    )
}

fn not_eq_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.is_empty() {
        return EvalResult::Err("Need at least one argument".into());
    }
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env) {
            EvalResult::Expr(expr) => Ok(expr),
            _ => Err("Can only test equality on numbers and symbols".into())
        })
        .collect::<Result<Vec<Rc<Expr>>, String>>();
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| {
            for i in 1..xs.len() {
                if xs[i] != xs[i-1] {
                    return EvalResult::Expr(Expr::symbol("True"));
                }
            }
            EvalResult::Expr(Expr::symbol("False"))
        }
    )
}

fn or_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.is_empty() {
        return EvalResult::Err("Need at least one argument".into());
    }
    let bools = vals.iter()
    .map(|e| match eval(e.clone(), env) {
        EvalResult::Expr(expr) => match &*expr {
            Expr::Symbol(s) => {
                if *s == "True" {
                    Ok(true)
                } else if *s == "False" {
                    Ok(false)
                } else {
                    Err("Can only or bools".into())
                }
            },
            Expr::List(l) => match l.len() {
                0 => Ok(false),
                1 => {
                    if *l[0] == Expr::FNum(1.0) {
                        Ok(true)
                    } else {
                        Err("Can only or bools".into())
                    }
                },
                _ => Err("Can only or bools".into())
            }
            _ => Err("Can only or bools".into()),
        },
        _ => Err("Can only or bools".into()),
    })
        .collect::<Result<Vec<bool>, String>>();
    bools.map_or_else(
        |err| EvalResult::Err(err),
        |bs| {
            let first = bs[0];
            match bs[1..].iter().fold(first, |acc, b| acc || *b) {
                true => EvalResult::Expr(Expr::symbol("True")),
                false => EvalResult::Expr(Expr::symbol("False")),
            }
        }
    )
}

fn and_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.is_empty() {
        return EvalResult::Err("Need at least one argument".into());
    }
    let bools = vals.iter()
        .map(|e| match eval(e.clone(), env) {
            EvalResult::Expr(expr) => match &*expr {
                Expr::Symbol(s) => {
                    if *s == "True" {
                        Ok(true)
                    } else if *s == "False" {
                        Ok(false)
                    } else {
                        Err("Can only and bools".into())
                    }
                },
                Expr::List(l) => match l.len() {
                    0 =>  Ok(false),
                    1 => {
                        if *l[0] == Expr::FNum(1.0) {
                            Ok(true)
                        } else {
                            Err("Can only and bools".into())
                        }
                    },
                    _ => Err("Can only and bools".into())
                }
                _ => Err("Can only and bools".into()),
            },
            _ => Err("Can only and bools".into()),
        })
        .collect::<Result<Vec<bool>, String>>();
    bools.map_or_else(
        |err| EvalResult::Err(err),
        |bs| {
            let first = bs[0];
            match bs[1..].iter().fold(first, |acc, b| acc && *b) {
                true => EvalResult::Expr(Expr::symbol("True")),
                false => EvalResult::Expr(Expr::symbol("False")),
            }
        }
    )
}

fn not_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.is_empty() {
        return EvalResult::Err("Need at least one argument".into());
    }
    if vals.len() == 1 {
        let e = &vals[0];
        let result = match eval(e.clone(), env) {
            EvalResult::Expr(expr) => match &*expr {
                Expr::List(l) => match l.len() {
                    0 => EvalResult::Expr(Expr::symbol("True")),
                    1 => {
                        if *l[0] == Expr::FNum(1.0) {
                            EvalResult::Expr(Expr::symbol("False"))
                        } else {
                            EvalResult::Err("Can only not bools".into())
                        }
                    },
                    _ => EvalResult::Err("Can only not bools".into())
                }
                Expr::Symbol(s) => {
                    if *s == "True" {
                        EvalResult::Expr(Expr::symbol("False"))
                    } else if *s == "False" {
                        EvalResult::Expr(Expr::symbol("True"))
                    } else {
                        EvalResult::Err("Can only not bools".into())
                    }
                },
                _ => EvalResult::Err("Can only not bools".into()),
            },
            _ => EvalResult::Err("Can only not bools".into()),
        };
        return result;
    } else {
        return EvalResult::Err("Not can only be applied to one expression".into());
    }
}

fn if_then_else(blocks: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if blocks.len() != 3 {
        return EvalResult::Err("If expression needs 3 arguments".into());
    }

    match eval(blocks[0].clone(), env) {
        EvalResult::Expr(expr) => {
            match &*expr {
                Expr::Symbol(s) => {
                    if *s == "True" {
                        eval(blocks[1].clone(), env)
                    } else if *s == "False" {
                        eval(blocks[2].clone(), env)
                    } else {
                        EvalResult::Err("If condition must evaluate to a bool".into())
                    }
                },
                Expr::List(vs) if vs.len() == 0 => eval(blocks[2].clone(), env),
                _ => eval(blocks[1].clone(), env),
            }
        },
        EvalResult::Unit => EvalResult::Err("If predicate cannot result in Unit".into()),
        err => err,
    }
}

/// Evaluates the given expression.
pub fn eval(e: Rc<Expr>, env: &mut Environment) -> EvalResult {
    match &*e {
        Expr::FNum(_) => EvalResult::Expr(e.clone()),
        Expr::Symbol(s) => evaluate_symbol(e.clone(), s, &[], env),
        Expr::List(vals) => {
            if vals.is_empty() {
                return EvalResult::Expr(Expr::list(&[]));
            }
            let op = &*vals[0];
            match op {
                Expr::Symbol(s) if s == "+" => add_vals(&vals[1..], env),
                Expr::Symbol(s) if s == "-" => sub_vals(&vals[1..], env),
                Expr::Symbol(s) if s == "*" => mul_vals(&vals[1..], env),
                Expr::Symbol(s) if s == "/" => div_vals(&vals[1..], env),
                Expr::Symbol(s) if s == "=" => eq_vals(&vals[1..], env),
                Expr::Symbol(s) if s == "!=" => not_eq_vals(&vals[1..], env),
                Expr::Symbol(s) if s == "or" => or_vals(&vals[1..], env),
                Expr::Symbol(s) if s == "and" => and_vals(&vals[1..], env),
                Expr::Symbol(s) if s == "not" => not_vals(&vals[1..], env),
                Expr::Symbol(s) if s == "let" => add_var_to_env(&vals[1..], env),
                Expr::Symbol(s) if s == "fn" => add_fn_to_env(&vals[1..], env),
                Expr::Symbol(s) if s == "if" => if_then_else(&vals[1..], env),
                Expr::Symbol(s) if s == "print" => {
                    let output: Vec<String> = vals[1..].iter()
                        .cloned()
                        .map(|expr| gen_print_output(expr, env))
                        .collect();
                    println!("{}", output.join(" "));
                    EvalResult::Unit
                },
                Expr::Symbol(s) if env.contains_key(&s) => evaluate_symbol(e.clone(), s, &vals[1..], env),
                _ => {
                    let res: Result<Vec<Rc<Expr>>, EvalResult> = vals.iter()
                    .cloned()
                    .map(|expr| eval(expr, env))
                    .filter(|x| *x != EvalResult::Unit)
                    .map(|x| if let EvalResult::Expr(expr) = x {
                        Ok(expr)
                    } else {
                        Err(x)
                    })
                    .collect();
                    res.map_or_else(
                        |err| err,
                        |exprs| EvalResult::Expr(Expr::list(&exprs)),
                    )
                },
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
