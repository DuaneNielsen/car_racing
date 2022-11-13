from cma_es import FastCovarianceMatrixAdaptation
from car_pid_control import Evaluator as CarRacingEvaluator
from racer_pid_control import Evaluator as RacerEvaluator
from car_path_pid_control import PIDControlEvaluator
import torch

cma = FastCovarianceMatrixAdaptation(8, step_mode='decay', step_decay=1e-2, samples=100)
evaluator = PIDControlEvaluator(num_runs=10, fps=100, render=False, device='cuda', crash_penalty=5., max_episode_steps=200)
demo_eval = PIDControlEvaluator(num_runs=1, fps=20, render=True, device='cuda',max_episode_steps=200)

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
        demo_eval(torch.stack([result['parameters'] for result in ranked_results[0:10]]))

    f.write('**************************************************************\n')
    f.write('                          END RUN                           \n')
    f.write('**************************************************************\n')