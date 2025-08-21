# DroneReinforcementLearning

## Overview

This project demonstrates reinforcement learning applied to drone control using AirSim, OpenAI Gym, and Stable Baselines3. It also includes a secondary reinforcement learning example with matrix visualization. 

**Before you begin:**  
Watch this introductory video for an overview: [Project Video](https://drive.google.com/file/d/1hm-66B-x_YJiRuOZaEUI84YJ-Y1KLDAO/view?usp=sharing)

Automatically generated code documentation can be found at:  
`doc/_build/html/index.html`

## Prerequisites

Install the following Python packages:

```bash
pip install numpy
pip install gym
pip install stable-baselines3[extra]
# For Mac users:
pip install 'stable-baselines3[extra]'
```

## Running the Drone Reinforcement Learning Code

1. Open [this Google Colab notebook](https://colab.research.google.com/drive/1-P9xrSE-l14V0mS0qsHiY_ZmHGxPqx01?usp=sharing).
2. Run all cells from top to bottom.
3. Once complete, open the folder icon in Colab. If you don't see `PPO.zip`, refresh the file list and download it.
4. [Install AirSim](https://microsoft.github.io/AirSim/build_macos/) on your computer following the "Host machine" instructions.
5. Open AirSim (see instructions under "How to Use AirSim").
6. In AirSim, click the dropdown with the double arrow next to "Compile", then click "Play".

    ![Cover Photo](/images/dropdown.png)

7. When prompted "Would you like to use car simulation?", select **No** to use quadrotor (drone) simulation.
8. You should now see a drone in AirSim.
9. Download `AirSimRL.py` from this repository and save it to `/AirSim/PythonClient/multirotor`.
10. Upload `PPO.zip` to the same folder.
11. Open a terminal and navigate to `/AirSim/PythonClient/multirotor`.
12. Run the script:

    ```bash
    python AirSimRL.py
    ```

13. Follow the terminal prompts. The drone should take off and fly autonomously using reinforcement learning.

## Running the Second Project (Matrix RL Example)

1. Open [this Colab notebook](https://colab.research.google.com/drive/1u7PHpb2CAS7DMYpy9sBmvOhdYCzVwZuC?usp=sharing).
2. Run all cells from top to bottom.
3. The output below the second to last cell will display two matrices:
    - The first matrix shows the grid layout, obstacle positions, start and goal.
    - The second matrix shows the reward values for each cell, with lower rewards near obstacles.

## Datasets

No external datasets were used; all experiments utilize reinforcement learning with simulated environments.

## Notes

- For troubleshooting AirSim installation and usage, refer to the official [AirSim documentation](https://microsoft.github.io/AirSim/build_macos/).
- For further code details, see the auto-generated docs at `doc/_build/html/index.html`.

---

Feel free to raise issues or contribute via pull requests!
