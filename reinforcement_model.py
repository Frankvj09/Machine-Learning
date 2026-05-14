import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import random

# Parámetros
EPISODES = 100

# Recompensas por episodio
rewards_per_episode = []

# Simulación simple
for episode in range(EPISODES):

    total_reward = 0

    # Simular decisiones
    for step in range(20):

        action = random.choice([0, 1])

        # recompensa
        if action == 1:
            reward = random.randint(5, 10)
        else:
            reward = random.randint(-5, 5)

        total_reward += reward

    rewards_per_episode.append(total_reward)

# Crear gráfica
def generar_grafica_rl():

    plt.figure(figsize=(8,5))

    plt.plot(rewards_per_episode)

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Reinforcement Learning Rewards")

    ruta = "static/reinforcement_plot.png"

    plt.savefig(ruta)

    plt.close()

    promedio = np.mean(rewards_per_episode)

    return promedio, ruta