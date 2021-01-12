import traci
import numpy as np
import random
import timeit
import os

# phase codes based on incrocio_prova.net.xml, the actions are intended as put green phase of the traffic light, so we have two actions : NS Green, EW Green
PHASE_NS_GREEN = 0  # Action 0
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2 # Action 1
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # Action 2
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6 # Action 3
PHASE_EWL_YELLOW = 7



class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []


    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        old_total_wait = 0
        old_action = -1 # dummy init

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            old_total_wait = current_total_wait

            self._reward_episode.append(reward)

        #print("Total reward:", np.sum(self._reward_episode))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time


    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length() 
            self._queue_length_episode.append(queue_length)


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["East2TrafficLight", "North2TrafficLight", "West2TrafficLight", "South2TrafficLight"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        return np.argmax(self._Model.predict_one(state))


    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TrafficLight", yellow_phase_code)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
    
        if action_number == 0:
            traci.trafficlight.setPhase("TrafficLight", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TrafficLight", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TrafficLight", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TrafficLight", PHASE_EWL_GREEN)


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("North2TrafficLight")
        halt_S = traci.edge.getLastStepHaltingNumber("South2TrafficLight")
        halt_E = traci.edge.getLastStepHaltingNumber("East2TrafficLight")
        halt_W = traci.edge.getLastStepHaltingNumber("West2TrafficLight")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length


    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros((8,34,3), dtype=np.float32)
        car_list = traci.vehicle.getIDList()
        cell_vel_time = np.zeros((8,34,3), dtype=np.float32)
        velocities = np.zeros((272), dtype=np.float32)
        times = np.zeros((272), dtype=np.float32)
        lane_cell = 0

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 799 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road
            flag = True
            initial_lane_pos = 6
            current_lane_pos = initial_lane_pos
            count = 0
            while flag == True and current_lane_pos <= 800:
                if lane_pos < current_lane_pos:
                    lane_cell = count
                    flag = False
                else: 
                    count += 1
                    current_lane_pos += initial_lane_pos + count

            # finding the lane where the car is located 
            if lane_id == "West2TrafficLight_0" or lane_id == "West2TrafficLight_1" or lane_id == "West2TrafficLight_2":
                lane_group = 0
            elif lane_id == "West2TrafficLight_3":
                lane_group = 1
            elif lane_id == "North2TrafficLight_0" or lane_id == "North2TrafficLight_1" or lane_id == "North2TrafficLight_2":
                lane_group = 2
            elif lane_id == "North2TrafficLight_3":
                lane_group = 3
            elif lane_id == "East2TrafficLight_0" or lane_id == "East2TrafficLight_1" or lane_id == "East2TrafficLight_2":
                lane_group = 4
            elif lane_id == "East2TrafficLight_3":
                lane_group = 5
            elif lane_id == "South2TrafficLight_0" or lane_id == "South2TrafficLight_1" or lane_id == "South2TrafficLight_2":
                lane_group = 6
            elif lane_id == "South2TrafficLight_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 0 and lane_group <= 7:
                #car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True

                cell_vel_time[lane_group][lane_cell][0] += 1
                cell_vel_time[lane_group][lane_cell][1] += traci.vehicle.getSpeed(car_id)
                cell_vel_time[lane_group][lane_cell][2] += traci.vehicle.getAccumulatedWaitingTime(car_id)
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

        for i in range (0, 8):
            for j in range (0, 34):
                if cell_vel_time[i][j][0] > 0:
                    # ci sono macchine nella cella
                    state[i][j][0] = 1 
                    if cell_vel_time[i][j][1] > 0 :
                        # somma delle velocità di tutti veicoli maggiore di 0
                        state[i][j][1] = cell_vel_time[i][j][1] / cell_vel_time[i][j][0]
                        velocities[(i * 34) + j] = state[i][j][1]
                        # media del tempo di attesa totale cumulato fino a quella cella
                        state[i][j][2] = cell_vel_time[i][j][2] / cell_vel_time[i][j][0]
                        times[(i * 34) + j] = state[i][j][2]
                    else:
                        # somma delle velocità di tutti veicoli uguale a 0
                        state[i][j][1] = 0 # There are cars but they are stuck
                        # media del tempo di attesa totale cumulato fino a quella cella 
                        # è presente anche se la velocità è 0
                        state[i][j][2] = cell_vel_time[i][j][2] / cell_vel_time[i][j][0]
                        times[(i * 34) + j] = state[i][j][2]
                else:
                    # non ci sono macchine nella cella, tutti i valori sono a 0
                    state[i][j][0] = 0
                    state[i][j][1] = 0
                    state[i][j][2] = 0
        
        if velocities.max(axis=0) != 0:
            normalized_vel = (velocities - velocities.min(axis=0)) / (velocities.max(axis=0) - velocities.min(axis=0))
            for i in range (0, 8):
                for j in range (0, 34):
                    state[i][j][1] = normalized_vel[(i * 34) + j]

        if times.max(axis=0) != 0:
            normalized_times = (times - times.min(axis=0)) / (times.max(axis=0) - times.min(axis=0))
            for i in range (0, 8):
                for j in range (0, 34):   
                    state[i][j][2] = normalized_times[(i * 34) + j]
        return state.flatten()



    @property
    def queue_length_episode(self):
        return self._queue_length_episode


    @property
    def reward_episode(self):
        return self._reward_episode



