from cma_es import FastCovarianceMatrixAdaptation
from car_pid_control import Evaluator as CarRacingEvaluator
from racer_pid_control import Evaluator as RacerEvaluator
from car_path_pid_control import PIDControlEvaluator

cma = FastCovarianceMatrixAdaptation(8, step_mode='decay', step_decay=1e-2, samples=100)
evaluator = PIDControlEvaluator(num_runs=10, fps=100, render=False, device='cuda')

with open('run_cma.txt', 'a') as f:
    f.write('**************************************************************\n')
    f.write('                          START RUN                           \n')
    f.write('**************************************************************\n')
    for step in cma.recommended_steps:
        f.write('**************************************************************\n')
        f.write(f'                          STEP {step}                          \n')
        f.write('**************************************************************\n')
        print(f'STEP {step}')
        ranked_results, info = cma.step(evaluator)

        for result in ranked_results[0:5]:
            fitness, parameters = result['fitness'], result['parameters']
            print(fitness, parameters)
            f.write(f'{fitness:.2f}, {parameters}\n')

    f.write('**************************************************************\n')
    f.write('                          END RUN                           \n')
    f.write('**************************************************************\n')