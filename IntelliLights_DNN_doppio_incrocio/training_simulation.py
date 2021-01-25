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
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Model_1 = Model[0]
        self._Model_2 = Model[1]
        self._Memory_1 = Memory[0]
        self._Memory_2 = Memory[1]
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store_1 = []
        self._reward_store_2 = []
        self._cumulative_wait_store_1 = []
        self._cumulative_wait_store_2 = []
        self._avg_queue_length_store_1 = []
        self._avg_queue_length_store_2 = []
        self._avg_speed_store_1 = []
        self._avg_speed_store_2 = []
        self._training_epochs = training_epochs


    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times_1 = {}
        self._waiting_times_2 = {}
        self._sum_neg_reward_1 = 0
        self._sum_neg_reward_2 = 0
        self._sum_queue_length_1 = 0
        self._sum_queue_length_2 = 0
        self._sum_waiting_time_1 = 0
        self._sum_waiting_time_2 = 0
        self._sum_avg_speed_1 = 0
        self._sum_avg_speed_2 = 0
        old_total_wait_1 = 0
        old_total_wait_2 = 0
        old_state_1 = -1
        old_state_2 = -1
        old_action_1 = -1
        old_action_2 = -1

        while self._step < self._max_steps:
            # get current state of the intersection
            current_state_1 = self._get_state_1()
            current_state_2 = self._get_state_2()
            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait_1 = self._collect_waiting_times_1()
            current_total_wait_2 = self._collect_waiting_times_2()
            reward_1 = old_total_wait_1 - current_total_wait_1
            reward_2 = old_total_wait_2 - current_total_wait_2

            # saving the data into the memory
            if self._step != 0:
                self._Memory_1.add_sample((old_state_1, old_action_1, reward_1, current_state_1))
                self._Memory_2.add_sample((old_state_2, old_action_2, reward_2, current_state_2))

            # choose the light phase to activate, based on the current state of the intersection
            action_1 = self._choose_action_1(current_state_1, epsilon)
            action_2 = self._choose_action_2(current_state_2, epsilon)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and (old_action_1 != action_1 or old_action_2 != action_2):
                if old_action_1 != action_1 and old_action_2 != action_2:
                    self._set_yellow_phase_1(old_action_1)
                    self._set_yellow_phase_2(old_action_2)
                    self._simulate(self._yellow_duration)
                elif old_action_1 != action_1:
                    self._set_yellow_phase_1(old_action_1)
                    self._simulate(self._yellow_duration)
                elif old_action_2 != action_2:
                    self._set_yellow_phase_2(old_action_2)
                    self._simulate(self._yellow_duration)


            # execute the phase selected before
            self._set_green_phase_1(action_1)
            self._set_green_phase_2(action_2)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state_1 = current_state_1
            old_state_2 = current_state_2
            old_action_1 = action_1
            old_action_2 = action_2
            old_total_wait_1 = current_total_wait_1
            old_total_wait_2 = current_total_wait_2

            # saving only the meaningful reward to better see if the agent is behaving correctly
            # Negative reward incentivize to reach terminal state as quick as possible
            if reward_1 < 0:
                self._sum_neg_reward_1 += reward_1
            if reward_2 < 0:
                self._sum_neg_reward_2 += reward_2 


        self._save_episode_stats()

        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print('Training...')
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time


    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        tot_steps = steps_todo
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length_1 = self._get_queue_length_1()
            queue_length_2 = self._get_queue_length_2()
            avg_speed_1 = self._get_avg_speed_1()
            avg_speed_2 = self._get_avg_speed_2()

            self._sum_queue_length_1 += queue_length_1
            self._sum_queue_length_2 += queue_length_2

            self._sum_waiting_time_1 += queue_length_1 # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds
            self._sum_waiting_time_2 += queue_length_2
            
            self._sum_avg_speed_1 = self._sum_avg_speed_1 + avg_speed_1
            self._sum_avg_speed_2 = self._sum_avg_speed_2 + avg_speed_2

        self._sum_avg_speed_1 = self._sum_avg_speed_1 / tot_steps
        self._sum_avg_speed_2 = self._sum_avg_speed_2 / tot_steps

    def _collect_waiting_times_1(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["East2TrafficLight_01", "North2TrafficLight_01", "West2TrafficLight_01", "South2TrafficLight_01"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times_1[car_id] = wait_time
            else:
                if car_id in self._waiting_times_1: # a car that was tracked has cleared the intersection
                    del self._waiting_times_1[car_id] 
        total_waiting_time = sum(self._waiting_times_1.values())
        return total_waiting_time

    def _collect_waiting_times_2(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["East2TrafficLight_02", "North2TrafficLight_02", "West2TrafficLight_02", "South2TrafficLight_02"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times_2[car_id] = wait_time
            else:
                if car_id in self._waiting_times_2: # a car that was tracked has cleared the intersection
                    del self._waiting_times_2[car_id] 
        total_waiting_time = sum(self._waiting_times_2.values())
        return total_waiting_time

    def _collect_avg_speed(self):
        """
        Retrieve the total average speed on the incoming roads
        """
        incoming_roads = ["East2TrafficLight_01", "North2TrafficLight_01", "WE2TrafficLight_01", "South2TrafficLight_01"]
        car_list = traci.vehicle.getIDList()
        speed = []
        for car_id in car_list:
            road_id = traci.vehicle.getRoadID(car_id)
            if(road_id in incoming_roads):
                speed.append(traci.vehicle.getSpeed(car_id))
        
        if(len(speed) > 0):
            return sum(speed)/len(speed)
        else:
            return 0

    def _collect_avg_speed_2(self):
        """
        Retrieve the total average speed on the incoming roads
        """
        incoming_roads = ["East2Traffighlight_02", "North2TrafficLight_02", "WE2TrafficLight_02", "South2TrafficLight_02"]
        car_list = traci.vehicle.getIDList()
        speed = []
        for car_id in car_list:
            road_id = traci.vehicle.getRoadID(car_id)
            if(road_id in incoming_roads):
                speed.append(traci.vehicle.getSpeed(car_id))
        
        if(len(speed) > 0):
            return sum(speed)/len(speed)
        else:
            return 0

    def _choose_action_1(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        Returns the indices of the maximum values along an axis.
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action, exploration
        else:
            return np.argmax(self._Model_1.predict_one(state)) # the best action given the current state, 

    def _choose_action_2(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        Returns the indices of the maximum values along an axis.
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action, exploration
        else:
            return np.argmax(self._Model_2.predict_one(state)) # the best action given the current state, 


    def _set_yellow_phase_1(self, old_action_1):
        """
        Activate the correct yellow light combination in sumo to the 1st TL
        """
        yellow_phase_code = old_action_1 * 2  + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        #print("Yellow phase code : " + str(yellow_phase_code))
        traci.trafficlight.setPhase("TrafficLight_01", yellow_phase_code)

    def _set_yellow_phase_2(self, old_action_2):
        """
        Activate the correct yellow light combination in sumo to the 2nd TL
        """
        yellow_phase_code = old_action_2 * 2  + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        #print("Yellow phase code : " + str(yellow_phase_code))
        traci.trafficlight.setPhase("TrafficLight_02", yellow_phase_code)


    def _set_green_phase_1(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TrafficLight_01", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TrafficLight_01", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TrafficLight_01", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TrafficLight_01", PHASE_EWL_GREEN)

    def _set_green_phase_2(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TrafficLight_02", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TrafficLight_02", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TrafficLight_02", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TrafficLight_02", PHASE_EWL_GREEN)



    def _get_queue_length_1(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("North2TrafficLight_01")
        halt_S = traci.edge.getLastStepHaltingNumber("South2TrafficLight_01")
        halt_E = traci.edge.getLastStepHaltingNumber("East2TrafficLight_01")
        halt_W = traci.edge.getLastStepHaltingNumber("West2TrafficLight_01")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _get_queue_length_2(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("North2TrafficLight_02")
        halt_S = traci.edge.getLastStepHaltingNumber("South2TrafficLight_02")
        halt_E = traci.edge.getLastStepHaltingNumber("East2TrafficLight_02")
        halt_W = traci.edge.getLastStepHaltingNumber("West2TrafficLight_02")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _get_avg_speed_1(self):
        """
        Retrieve the avg speed of cars withing the incoming lane
        """
        avg_N = traci.edge.getLastStepMeanSpeed("North2TrafficLight_01")
        avg_S = traci.edge.getLastStepMeanSpeed("South2TrafficLight_01")
        avg_E = traci.edge.getLastStepMeanSpeed("East2TrafficLight_01")
        avg_W = traci.edge.getLastStepMeanSpeed("West2TrafficLight_01")
        avg_total = (avg_N + avg_S + avg_E + avg_W) / 4 # Average global speed incoming into the tl
        return avg_total

    def _get_avg_speed_2(self):
        """
        Retrieve the avg speed of cars withing the incoming lane
        """
        avg_N = traci.edge.getLastStepMeanSpeed("North2TrafficLight_02")
        avg_S = traci.edge.getLastStepMeanSpeed("South2TrafficLight_02")
        avg_E = traci.edge.getLastStepMeanSpeed("East2TrafficLight_02")
        avg_W = traci.edge.getLastStepMeanSpeed("West2TrafficLight_02")
        avg_total = (avg_N + avg_S + avg_E + avg_W) / 4 # Average global speed incoming into the tl
        return avg_total


    def _get_state_1(self):
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
            if lane_id == "West2TrafficLight_01_0" or lane_id == "West2TrafficLight_01_1" or lane_id == "West2TrafficLight_01_2":
                lane_group = 0
            elif lane_id == "West2TrafficLight_01_3":
                lane_group = 1
            elif lane_id == "North2TrafficLight_01_0" or lane_id == "North2TrafficLight_01_1" or lane_id == "North2TrafficLight_01_2":
                lane_group = 2
            elif lane_id == "North2TrafficLight_01_3":
                lane_group = 3
            elif lane_id == "East2TrafficLight_01_0" or lane_id == "East2TrafficLight_01_1" or lane_id == "East2TrafficLight_01_2":
                lane_group = 4
            elif lane_id == "East2TrafficLight_01_3":
                lane_group = 5
            elif lane_id == "South2TrafficLight_01_0" or lane_id == "South2TrafficLight_01_1" or lane_id == "South2TrafficLight_01_2":
                lane_group = 6
            elif lane_id == "South2TrafficLight_01_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 0 and lane_group <= 7:
                #car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
                cell_vel_time[lane_group][lane_cell][0] += 1
                cell_vel_time[lane_group][lane_cell][1] += traci.vehicle.getSpeed(car_id)
                cell_vel_time[lane_group][lane_cell][2] += traci.vehicle.getAccumulatedWaitingTime(car_id)
            elif lane_group == 0:
                #car_position = lane_cell
                valid_car = True
                cell_vel_time[lane_group][lane_cell][0] += 1
                cell_vel_time[lane_group][lane_cell][1] += traci.vehicle.getSpeed(car_id)
                cell_vel_time[lane_group][lane_cell][2] += traci.vehicle.getAccumulatedWaitingTime(car_id)
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

        for i in range (0, 8):
            for j in range (0,34):
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

    def _get_state_2(self):
        """
        Retrieve the state of the first intersection from sumo, in the form of cell occupancy
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
            if lane_id == "West2TrafficLight_02_0" or lane_id == "West2TrafficLight_02_1" or lane_id == "West2TrafficLight_02_2":
                lane_group = 0
            elif lane_id == "West2TrafficLight_02_3":
                lane_group = 1
            elif lane_id == "North2TrafficLight_02_0" or lane_id == "North2TrafficLight_02_1" or lane_id == "North2TrafficLight_02_2":
                lane_group = 2
            elif lane_id == "North2TrafficLight_02_3":
                lane_group = 3
            elif lane_id == "East2TrafficLight_02_0" or lane_id == "East2TrafficLight_02_1" or lane_id == "East2TrafficLight_02_2":
                lane_group = 4
            elif lane_id == "East2TrafficLight_02_3":
                lane_group = 5
            elif lane_id == "South2TrafficLight_02_0" or lane_id == "South2TrafficLight_02_1" or lane_id == "South2TrafficLight_02_2":
                lane_group = 6
            elif lane_id == "South2TrafficLight_02_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 0 and lane_group <= 7:
                #car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True

                cell_vel_time[lane_group][lane_cell][0] += 1
                cell_vel_time[lane_group][lane_cell][1] += traci.vehicle.getSpeed(car_id)
                cell_vel_time[lane_group][lane_cell][2] += traci.vehicle.getAccumulatedWaitingTime(car_id)
            elif lane_group == 0:
                #car_position = lane_cell
                valid_car = True
                cell_vel_time[lane_group][lane_cell][0] += 1
                cell_vel_time[lane_group][lane_cell][1] += traci.vehicle.getSpeed(car_id)
                cell_vel_time[lane_group][lane_cell][2] += traci.vehicle.getAccumulatedWaitingTime(car_id)
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

        for i in range (0, 8):
            for j in range (0,34):
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

    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch_1 = self._Memory_1.get_samples(self._Model_1.batch_size)
        batch_2 = self._Memory_2.get_samples(self._Model_2.batch_size)
        if len(batch_1) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch_1])  # extract states from the batch_1
            next_states = np.array([val[3] for val in batch_1])  # extract next states from the batch_1

            # prediction
            q_s_a = self._Model_1.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model_1.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch_1), 816))
            y = np.zeros((len(batch_1), self._num_actions))

            for i, b in enumerate(batch_1):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                # amax return the max value along the axis
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value
            self._Model_1.train_batch(x, y)  # train the NN
        if len(batch_2) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch_2])  # extract states from the batch_2
            next_states = np.array([val[3] for val in batch_2])  # extract next states from the batch_2

            # prediction
            q_s_a = self._Model_2.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model_2.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch_2), 816))
            y = np.zeros((len(batch_2), self._num_actions))

            for i, b in enumerate(batch_2):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                # amax return the max value along the axis
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value
            self._Model_2.train_batch(x, y)  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store_1.append(self._sum_neg_reward_1)  # how much negative reward in this episode
        self._reward_store_2.append(self._sum_neg_reward_2)  # how much negative reward in this episode
        self._cumulative_wait_store_1.append(self._sum_waiting_time_1)  # total number of seconds waited by cars in this episode
        self._cumulative_wait_store_2.append(self._sum_waiting_time_2)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store_1.append(self._sum_queue_length_1 / self._max_steps)  # average number of queued cars per step, in this episode
        self._avg_queue_length_store_2.append(self._sum_queue_length_2 / self._max_steps)  # average number of queued cars per step, in this episode
        self._avg_speed_store_1.append(self._sum_avg_speed_1)
        self._avg_speed_store_2.append(self._sum_avg_speed_2)

    @property
    def reward_store_1(self):
        return self._reward_store_1

    @property
    def reward_store_2(self):
        return self._reward_store_2


    @property
    def cumulative_wait_store_1(self):
        return self._cumulative_wait_store_1

    @property
    def cumulative_wait_store_2(self):
        return self._cumulative_wait_store_2

    @property
    def avg_queue_length_store_1(self):
        return self._avg_queue_length_store_1

    @property
    def avg_queue_length_store_2(self):
        return self._avg_queue_length_store_2

    @property
    def avg_speed_store_1(self):
        return self._avg_speed_store_1

    @property
    def avg_speed_store_2(self):
        return self._avg_speed_store_2
