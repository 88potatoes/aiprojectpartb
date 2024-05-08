command to run

repeat 10 python3 -m referee -v 0 random_agent monte_carlo_agent >> results_rand_vs_monte_carlo.txt     
repeat 50 python3 -m referee -v 0 monte_carlo_strong_agent monte_carlo_agent >> str_vs_norm.txt
repeat 20 python3 -m referee -v 0 random_agent minimax_agent >> random_vs_minimax.txt 