use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{Search, Solution};
use std::error::Error;
use std::fmt::Display;
use std::fs::OpenOptions;
use std::io::Write;

/// Run a solver and dump the solution history to a CSV file.
///
/// The first field is the time, second is the cost, third is the bound, fourth is the transitions,
/// fifth is the expanded, and sixth is the generated.
pub fn run_solver_and_dump_solution_history<C>(
    solver: &mut Box<dyn Search<C>>,
    filename: &str,
) -> Result<Solution<C>, Box<dyn Error>>
where
    C: Numeric + Display + Copy,
{
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(filename)?;

    loop {
        let (solution, terminated) = solver.search_next()?;

        if let Some(cost) = solution.cost {
            let transitions = solution
                .transitions
                .iter()
                .map(|t| t.get_full_name())
                .collect::<Vec<_>>()
                .join(" ");

            let line = if let Some(bound) = solution.best_bound {
                format!(
                    "{}, {}, {}, {}, {}, {}\n",
                    solution.time, cost, bound, transitions, solution.expanded, solution.generated
                )
            } else {
                format!(
                    "{}, {}, , {}, {}, {}\n",
                    solution.time, cost, transitions, solution.expanded, solution.generated
                )
            };
            file.write_all(line.as_bytes())?;
            file.flush()?;
        }

        if terminated {
            return Ok(solution);
        }
    }
}

/// Print the cost, bound, and statistics of a solution.
pub fn print_solution_statistics<C>(solution: &Solution<C>)
where
    C: Numeric + Copy + Display,
{
    if let Some(cost) = solution.cost {
        println!("cost: {}", cost);

        if solution.is_optimal {
            println!("optimal cost: {}", cost);
        }
    } else {
        println!("No solution is found.");

        if solution.is_infeasible {
            println!("The problem is infeasible.");
        }
    }

    if let Some(bound) = solution.best_bound {
        println!("best bound: {}", bound);
    }

    println!("Search time: {}s", solution.time);
    println!("Expanded: {}", solution.expanded);
    println!("Generated: {}", solution.generated);
}
