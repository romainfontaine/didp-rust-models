use clap::Parser;
use fixedbitset::FixedBitSet;
use rpid::prelude::*;
use rpid::{algorithms, io, solvers, timer::Timer};
use std::cmp::{self, Ordering};
use tsptw::{Args, Instance, SimplificationChoice, SolverChoice};
use proc_status::ProcStatus;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

struct Tsptw {
    instance: Instance,
    c_star: Vec<Vec<Option<i32>>>,
    min_to: Vec<i32>,
    min_from: Vec<i32>,
    minimize_makespan: bool,
}

impl Tsptw {
    fn new(instance: Instance, minimize_makespan: bool) -> Self {
        let mut c = instance.c.clone();
        c.iter_mut().for_each(|row| {
            row[0] = None;
        });
        let c_star = algorithms::compute_pairwise_shortest_path_costs_with_option(&c);
        let min_to = algorithms::take_column_wise_min_with_option(&instance.c)
            .map(|x| x.unwrap())
            .collect();
        let min_from = algorithms::take_row_wise_min_with_option(&instance.c)
            .map(|x| x.unwrap())
            .collect();

        Self {
            instance,
            c_star,
            min_to,
            min_from,
            minimize_makespan,
        }
    }
}

struct TsptwState {
    unvisited: FixedBitSet,
    current: usize,
    time: i32,
}

impl Tsptw {
    fn check_feasibility(&self, state: &TsptwState) -> bool {
        state.unvisited.ones().all(|next| {
            if let Some(distance) = self.c_star[state.current][next] {
                state.time + distance <= self.instance.b[next]
            } else {
                false
            }
        })
    }
}

impl Dp for Tsptw {
    type State = TsptwState;
    type CostType = i32;

    fn get_target(&self) -> TsptwState {
        let mut unvisited = FixedBitSet::with_capacity(self.instance.a.len());
        unvisited.insert_range(1..);

        TsptwState {
            unvisited,
            current: 0,
            time: 0,
        }
    }

    fn get_successors(
        &self,
        state: &TsptwState,
    ) -> impl IntoIterator<Item = (TsptwState, i32, usize)> {
        if state.unvisited.is_clear() {
            let next = 0;
            if let Some(distance) = self.instance.c[state.current][next] {
                let time = state.time + distance;
                // (no waiting possible here)
                if time <= self.instance.b[next] {
                    let successor = TsptwState {
                        unvisited: state.unvisited.clone(),
                        current: next,
                        time,
                    };
                    // don't remove 'next' from unvisited
                    // don't 'self.check_feasibility(&successor)'
                    return vec![(successor, distance, next)];
                }
            }
            return vec![];
        }
        state.unvisited.ones().filter_map(|next| {
            if let Some(distance) = self.instance.c[state.current][next] {
                let time = cmp::max(state.time + distance, self.instance.a[next]);

                if time > self.instance.b[next] {
                    return None;
                }

                let mut unvisited = state.unvisited.clone();
                unvisited.remove(next);

                let successor = TsptwState {
                    unvisited,
                    current: next,
                    time,
                };

                if !self.check_feasibility(&successor) {
                    return None;
                }

                let weight = if self.minimize_makespan {
                    time - state.time
                } else {
                    distance
                };

                Some((successor, weight, next))
            } else {
                None
            }
        }).collect()
    }

    fn get_base_cost(&self, state: &TsptwState) -> Option<i32> {
        if state.unvisited.is_clear() && state.current == 0 {
            Some(0)
        } else {
            None
        }
    }
}

impl Dominance for Tsptw {
    type State = TsptwState;
    type Key = (FixedBitSet, usize);

    fn get_key(&self, state: &TsptwState) -> Self::Key {
        (state.unvisited.clone(), state.current)
    }

    fn compare(&self, a: &TsptwState, b: &Self::State) -> Option<Ordering> {
        Some(b.time.cmp(&a.time))
    }
}

impl Bound for Tsptw {
    type State = TsptwState;
    type CostType = i32;

    fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
        if state.unvisited.is_clear() {
            if state.current == 0 {
                return Some(0);
            } else {
                return self.instance.c[state.current][0];
            }
        }
        let bound_to = state.unvisited.ones().map(|i| self.min_to[i]).sum::<i32>() + self.min_to[0];
        let bound_from = self.min_from[state.current]
            + state
                .unvisited
                .ones()
                .map(|i| self.min_from[i])
                .sum::<i32>();

        Some(cmp::max(bound_to, bound_from))
    }
}

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let mut instance = Instance::read_from_file(&args.input_file).unwrap();

    match args.simplification_level {
        SimplificationChoice::None => {}
        SimplificationChoice::Cheap => {
            instance.simplify(false);
        }
        SimplificationChoice::Expensive => {
            instance.simplify(true);
        }
    }

    let tsptw = Tsptw::new(instance.clone(), args.minimize_makespan);

    let parameters = SearchParameters {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };
    let solution = match args.solver {
        SolverChoice::Cabs => {
            let cabs_parameters = CabsParameters::default();
            println!("Preparing time: {}s", timer.get_elapsed_time());
            let mut solver = solvers::create_cabs(tsptw, parameters, cabs_parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {}s", timer.get_elapsed_time());
            let mut solver = solvers::create_astar(tsptw, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
    };
    io::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        let transitions = solution
            .transitions
            .iter()
            .map(|t| format!("{}", t))
            .collect::<Vec<_>>()
            .join(" ");
        println!("Tour: {}", transitions);

        if (args.minimize_makespan && instance.validate_makespan(&solution.transitions, cost))
            || instance.validate(&solution.transitions, cost)
        {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
    let ps = ProcStatus::read().unwrap();
    println!("VmPeak: {}", ps.value_KiB("VmPeak").unwrap());
}
