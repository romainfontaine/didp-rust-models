use clap::Parser;
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    create_caasdy, create_dual_bound_cabs, BeamSearchParameters, CabsParameters, FEvaluatorType,
    Parameters,
};
use rpid::timer::Timer;
use std::rc::Rc;
use talent_scheduling::{Args, Instance, SolverChoice};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let instance = Instance::read_from_file(&args.input_file).unwrap();
    let (simplified_instance, single_actor_cost, scene_to_originals) = instance.simplify();

    let mut model = Model::default();

    let n = simplified_instance.scene_to_duration.len();
    let scene = model.add_object_type("scene", n).unwrap();
    let m = simplified_instance.actor_to_cost.len();
    let actor = model.add_object_type("actor", m).unwrap();

    let remaining = (0..n).collect::<Vec<_>>();
    let remaining = model.create_set(scene, &remaining).unwrap();
    let remaining = model
        .add_set_variable("remaining", scene, remaining)
        .unwrap();

    let scene_to_actors = simplified_instance.create_scene_to_actors();
    let mut scene_to_subsumption_candidates = vec![vec![]; n];

    for i in 0..n {
        for j in 0..n {
            if i != j && scene_to_actors[i].is_subset(&scene_to_actors[j]) {
                scene_to_subsumption_candidates[i].push(j);
            }
        }
    }

    let scene_to_base_cost = scene_to_actors
        .iter()
        .enumerate()
        .map(|(i, actors)| {
            actors
                .ones()
                .map(|j| simplified_instance.actor_to_cost[j])
                .sum::<i32>()
                * simplified_instance.scene_to_duration[i]
        })
        .collect();
    let scene_to_base_cost = model
        .add_table_1d("scene to base cost", scene_to_base_cost)
        .unwrap();
    let scene_to_actors = scene_to_actors
        .iter()
        .map(|actors| {
            let set = actors.ones().collect::<Vec<_>>();
            model.create_set(actor, &set).unwrap()
        })
        .collect::<Vec<_>>();
    let scene_to_actors = model
        .add_table_1d("scene to actors", scene_to_actors)
        .unwrap();
    let actor_cost = model
        .add_table_1d("actor cost", simplified_instance.actor_to_cost.clone())
        .unwrap();

    for i in 0..n {
        let standby = scene_to_actors.union(m, remaining) & scene_to_actors.union(m, !remaining);

        let mut actor_equivalent_shoot = Transition::new(format!("{}", i));
        actor_equivalent_shoot.set_cost(scene_to_base_cost.element(i) + IntegerExpression::Cost);
        actor_equivalent_shoot
            .add_effect(remaining, remaining.remove(i))
            .unwrap();
        actor_equivalent_shoot.add_precondition(remaining.contains(i));
        actor_equivalent_shoot
            .add_precondition(scene_to_actors.element(i).is_subset(standby.clone()));
        actor_equivalent_shoot.add_precondition(standby.is_subset(scene_to_actors.element(i)));

        model
            .add_forward_forced_transition(actor_equivalent_shoot)
            .unwrap();
    }

    for (i, &d) in simplified_instance.scene_to_duration.iter().enumerate() {
        let standby = scene_to_actors.union(m, remaining) & scene_to_actors.union(m, !remaining);
        let on_location = scene_to_actors.element(i) | standby;

        let mut shoot = Transition::new(format!("{}", i));
        shoot.set_cost(d * actor_cost.sum(on_location) + IntegerExpression::Cost);
        shoot.add_effect(remaining, remaining.remove(i)).unwrap();
        shoot.add_precondition(remaining.contains(i));

        for &j in &scene_to_subsumption_candidates[i] {
            shoot.add_precondition(
                !remaining.contains(j)
                    | !(scene_to_actors.element(j).is_subset(
                        scene_to_actors.union(m, !remaining) | scene_to_actors.element(i),
                    )),
            );
        }

        model.add_forward_transition(shoot).unwrap();
    }

    model.add_base_case(vec![remaining.is_empty()]).unwrap();

    model
        .add_dual_bound(scene_to_base_cost.sum(remaining))
        .unwrap();

    let model = Rc::new(model);

    let parameters = Parameters::<i32> {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };

    let mut solver = match args.solver {
        SolverChoice::Cabs => {
            let beam_search_parameters = BeamSearchParameters {
                parameters,
                ..Default::default()
            };
            let parameters = CabsParameters {
                beam_search_parameters,
                ..Default::default()
            };
            println!("Preparing time: {}s", timer.get_elapsed_time());

            create_dual_bound_cabs(model, parameters, FEvaluatorType::Plus)
        }
        SolverChoice::Astar => {
            println!("Preparing time: {}s", timer.get_elapsed_time());

            create_caasdy(model, parameters, FEvaluatorType::Plus)
        }
    };

    let solution =
        io_util::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap();
    io_util::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        let schedule = solution
            .transitions
            .iter()
            .map(|t| t.get_full_name().parse::<usize>().unwrap())
            .flat_map(|i| scene_to_originals[i].iter().cloned())
            .collect::<Vec<_>>();
        println!(
            "Schedule: {}",
            schedule
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
        let cost = cost + single_actor_cost;

        if instance.validate(&schedule, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
