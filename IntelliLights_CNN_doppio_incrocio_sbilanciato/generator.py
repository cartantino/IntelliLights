import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/routes.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="E_N_01" edges="East2TrafficLight_01 TrafficLight2North_01"/>
            <route id="E_S_01" edges="East2TrafficLight_01 TrafficLight2South_01"/>
            <route id="E_W_N_01" edges="East2TrafficLight_01 East2TrafficLight_02 TrafficLight2North_02"/>
            <route id="E_W_S_01" edges="East2TrafficLight_01 East2TrafficLight_02 TrafficLight2South_02"/>
            <route id="E_W_W_01" edges="East2TrafficLight_01 East2TrafficLight_02 TrafficLight2West_02"/>

            <route id="S_N_01" edges="South2TrafficLight_01 TrafficLight2North_01"/>
            <route id="S_E_01" edges="South2TrafficLight_01 TrafficLight2East_01"/>
            <route id="S_W_N_01" edges="South2TrafficLight_01 East2TrafficLight_02 TrafficLight2North_02"/>
            <route id="S_W_S_01" edges="South2TrafficLight_01 East2TrafficLight_02 TrafficLight2South_02"/>
            <route id="S_W_W_01" edges="South2TrafficLight_01 East2TrafficLight_02 TrafficLight2West_02"/>

            <route id="N_E_01" edges="North2TrafficLight_01 TrafficLight2East_01"/>
            <route id="N_S_01" edges="North2TrafficLight_01 TrafficLight2South_01"/>
            <route id="N_W_N_01" edges="North2TrafficLight_01 East2TrafficLight_02 TrafficLight2North_02"/>
            <route id="N_W_S_01" edges="North2TrafficLight_01 East2TrafficLight_02 TrafficLight2South_02"/>
            <route id="N_W_W_01" edges="North2TrafficLight_01 East2TrafficLight_02 TrafficLight2West_02"/>


            <route id="W_N_02" edges="West2TrafficLight_02 TrafficLight2North_02"/>
            <route id="W_S_02" edges="West2TrafficLight_02 TrafficLight2South_02"/>
            <route id="W_E_N_02" edges="West2TrafficLight_02 West2TrafficLight_01 TrafficLight2North_01"/>
            <route id="W_E_S_02" edges="West2TrafficLight_02 West2TrafficLight_01 TrafficLight2South_01"/>
            <route id="W_E_E_02" edges="West2TrafficLight_02 West2TrafficLight_01 TrafficLight2East_01"/>

            <route id="S_W_02" edges="South2TrafficLight_02 TrafficLight2West_02"/>
            <route id="S_N_02" edges="South2TrafficLight_02 TrafficLight2North_02"/>
            <route id="S_E_N_02" edges="South2TrafficLight_02 West2TrafficLight_01 TrafficLight2North_01"/>
            <route id="S_E_S_02" edges="South2TrafficLight_02 West2TrafficLight_01 TrafficLight2South_01"/>
            <route id="S_E_E_02" edges="South2TrafficLight_02 West2TrafficLight_01 TrafficLight2East_01"/>

            <route id="N_W_02" edges="North2TrafficLight_02 TrafficLight2West_02"/>
            <route id="N_S_02" edges="North2TrafficLight_02 TrafficLight2South_02"/>
            <route id="N_E_N_02" edges="North2TrafficLight_02 West2TrafficLight_01 TrafficLight2North_01"/>
            <route id="N_E_S_02" edges="North2TrafficLight_02 West2TrafficLight_01 TrafficLight2South_01"/>
            <route id="N_E_E_02" edges="North2TrafficLight_02 West2TrafficLight_01 TrafficLight2East_01"/>

            """, file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.85:  # choose direction: straight or turn - 85% of times the car goes straight
                    unbalanced = np.random.uniform()
                    if unbalanced < 0.85:
                        route_straight = np.random.randint(1, 3)  # 0.7225
                        if route_straight == 1:
                            print('    <vehicle id="E_W_W_01_%i" type="standard_car" route="E_W_W_01" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)                        
                        elif route_straight == 2:
                            print('    <vehicle id="W_E_E_02_%i" type="standard_car" route="W_E_E_02" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else:
                        route_straight = np.random.randint(1, 5) # 0.1275
                        if route_straight == 1:
                            print('    <vehicle id="N_S_02_%i" type="standard_car" route="N_S_02" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif route_straight == 2:
                            print('    <vehicle id="S_N_02_%i" type="standard_car" route="S_N_02" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif route_straight == 3:
                            print('    <vehicle id="N_S_01_%i" type="standard_car" route="N_S_01" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif route_straight == 4:
                            print('    <vehicle id="S_N_01_%i" type="standard_car" route="S_N_01" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

                else:
                    route_turn = np.random.randint(1, 25)  # 0.15
                    if route_turn == 1:
                        print('    <vehicle id="E_N_01_%i" type="standard_car" route="E_N_01" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="E_S_01_%i" type="standard_car" route="E_S_01" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 3:
                        print('    <vehicle id="E_W_N_01_%i" type="standard_car" route="E_W_N_01" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 4:
                        print('    <vehicle id="E_W_S_01_%i" type="standard_car" route="E_W_S_01" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

                    elif route_turn == 5:
                        print('    <vehicle id="S_E_01_%i" type="standard_car" route="S_E_01" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="S_W_N_01_%i" type="standard_car" route="S_W_N_01" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="S_W_S_01_%i" type="standard_car" route="S_W_S_01" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 8:
                        print('    <vehicle id="S_W_W_01_%i" type="standard_car" route="S_W_W_01" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    
                    elif route_turn == 9:
                        print('    <vehicle id="N_E_01_%i" type="standard_car" route="N_E_01" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 10:
                        print('    <vehicle id="N_W_N_01_%i" type="standard_car" route="N_W_N_01" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 11:
                        print('    <vehicle id="N_W_S_01_%i" type="standard_car" route="N_W_S_01" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 12:
                        print('    <vehicle id="N_W_W_01_%i" type="standard_car" route="N_W_W_01" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    
                    elif route_turn == 13:
                        print('    <vehicle id="W_N_02_%i" type="standard_car" route="W_N_02" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 14:
                        print('    <vehicle id="W_S_02_%i" type="standard_car" route="W_S_02" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 15:
                        print('    <vehicle id="W_E_N_02_%i" type="standard_car" route="W_E_N_02" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 16:
                        print('    <vehicle id="W_E_S_02_%i" type="standard_car" route="W_E_S_02" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

                    elif route_turn == 17:
                        print('    <vehicle id="S_W_02_%i" type="standard_car" route="S_W_02" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 18:
                        print('    <vehicle id="S_E_N_02_%i" type="standard_car" route="S_E_N_02" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 19:
                        print('    <vehicle id="S_E_S_02_%i" type="standard_car" route="S_E_S_02" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 20:
                        print('    <vehicle id="S_E_E_02_%i" type="standard_car" route="S_E_E_02" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

                    elif route_turn == 21:
                        print('    <vehicle id="N_W_02_%i" type="standard_car" route="N_W_02" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 22:
                        print('    <vehicle id="N_E_N_02_%i" type="standard_car" route="N_E_N_02" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 23:
                        print('    <vehicle id="N_E_S_02_%i" type="standard_car" route="N_E_S_02" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 24:
                        print('    <vehicle id="N_E_E_02_%i" type="standard_car" route="N_E_E_02" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)                                                
                
            print("</routes>", file=routes)