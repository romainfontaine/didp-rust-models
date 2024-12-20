use clap::{Parser, ValueEnum};
use fixedbitset::FixedBitSet;
use itertools::Itertools;

pub const KNOWN_OPTIMAL_COSTS: [usize; 29] = [
    0, 0, 1, 3, 6, 11, 17, 25, 34, 44, 55, 72, 85, 106, 127, 151, 177, 199, 216, 246, 283, 333,
    356, 372, 425, 480, 492, 553, 585,
];

pub fn validate(n: usize, marks: &[usize], length: usize) -> bool {
    if marks.len() != n {
        println!(
            "The number of marks {} is different from {}",
            marks.len(),
            n
        );

        return false;
    }

    let mut distance_set = FixedBitSet::with_capacity(n * n);
    let mut max_distance = 0;

    for (&i, &j) in marks.iter().tuple_combinations() {
        let distance = if i > j { i - j } else { j - i };

        if distance_set.contains(distance) {
            println!("Distance {} is repeated", distance);

            return false;
        }

        distance_set.insert(distance);

        if distance > max_distance {
            max_distance = distance;
        }
    }

    if max_distance != length {
        println!(
            "The maximum distance {} is different from length {}",
            max_distance, length
        );

        return false;
    }

    true
}

#[derive(Debug, Clone, ValueEnum)]
pub enum SolverChoice {
    Cabs,
    Astar,
}

#[derive(Debug, Parser)]
pub struct Args {
    #[arg(help = "n")]
    pub n: usize,
    #[arg(short, long, value_enum, default_value_t = SolverChoice::Cabs, help = "Solver")]
    pub solver: SolverChoice,
    #[arg(long, default_value_t = String::from("history.csv"), help = "File to save the history")]
    pub history: String,
    #[arg(short, long, default_value_t = 1800.0, help = "Time limit")]
    pub time_limit: f64,
}
