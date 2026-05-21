# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:20:28 2019

@author: Ritu Pande
"""
import os
import traci
import numpy as np
class SumoEnv:
    def __init__(self, sumoBinary, max_steps):
         # Ensure SUMO output directory exists to avoid "Could not build output file" errors
         os.makedirs("results", exist_ok=True)
         self.sumoCmd = [sumoBinary, "-c", "intersection/ts.4L.sumocfg", "--no-step-log", "true", "--waiting-time-memory", str(max_steps), "--log","results/logfile.txt"]
         self.SUMO_INT_LANE_LENGTH = 500
         self.num_states = 88 # 80 binary position cells + 8 normalized scalar features
         self.max_steps = max_steps
         # reward weights: delay, stops, queue, CO2
         self.w1 = 0.4
         self.w2 = 0.2
         self.w3 = 0.2
         self.w4 = 0.0001  # CO2 is in mg/s so needs a much smaller weight
         self._init()
                  
    def _init(self):
        self.current_state = None
        self.curr_wait_time = 0
        self.steps = 0
        self.last_queue_sum = 0
        self.last_delay_sum = 0
        self.last_stops_sum = 0
        self.last_co2_sum = 0
        self.last_speed_sum = 0
        self.last_vehicle_count_sum = 0
        
    def get_state(self):
        return self.current_state
    
    def start(self):
        try:
            traci.start(self.sumoCmd)
            self.current_state = self._encode_env_state()
            return self.current_state
        except Exception as e:
            print(f"SUMO start FAILED: {type(e).__name__}: {e}")
            raise

        
    def reset(self):
        try:
            traci.close()
            traci.start(self.sumoCmd)
            self._init()
            self.current_state = self._encode_env_state()
            return self.current_state
        except Exception as e:
            print(f"SUMO reset FAILED: {type(e).__name__}: {e}")
            raise

    
    def step( self, num_steps = 1 ):
        if self.steps + num_steps > self.max_steps:
            num_steps = self.max_steps - self.steps

        queue_sum = 0
        delay_sum = 0
        stops_sum = 0
        co2_sum = 0
        speed_sum = 0
        vehicle_count_sum = 0

        for i in range(num_steps):
            traci.simulationStep()
            queue_sum += self.get_intersection_q_per_step()
            new_wait = self._get_waiting_time()
            delay_sum += new_wait - self.curr_wait_time  # delta: increase in wait time this step
            self.curr_wait_time = new_wait
            stops_sum += self.get_stops_per_step()
            co2_sum += self.get_co2_per_step()
            speed_sum += self.get_average_speed_per_step()
            vehicle_count_sum += self.get_vehicle_count_per_step()

        self.last_queue_sum = queue_sum
        self.last_delay_sum = delay_sum
        self.last_stops_sum = stops_sum
        self.last_co2_sum = co2_sum
        self.last_speed_sum = speed_sum
        self.last_vehicle_count_sum = vehicle_count_sum

        self.steps += num_steps
        self.current_state = self._encode_env_state()
        # new_wait_time  = self._get_waiting_time()
        # #print("new_wait_time={}".format(new_wait_time))

        # # calculate reward of action taken (change in cumulative waiting time between actions)
        # # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
        # reward = self.curr_wait_time - new_wait_time
        # #print("reward={}".format(reward))
        # self.curr_wait_time = new_wait_time

        reward = -(
            self.w1 * delay_sum +
            self.w2 * stops_sum +
            self.w3 * queue_sum +
            self.w4 * co2_sum
        ) / max(1, num_steps)
        
        # one episode ends when all vehicles have arrived at their destination
        is_terminal = (
            self.steps >= self.max_steps
            or traci.simulation.getMinExpectedNumber() == 0
        )
            
        return (reward, self.current_state, is_terminal)
 
    # RETRIEVE THE WAITING TIME OF EVERY CAR IN THE INCOMING LANES
    def _get_waiting_time(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        total_waiting_time = 0
        for veh_id in traci.vehicle.getIDList():
            wait_time_car = traci.vehicle.getAccumulatedWaitingTime(veh_id)
            road_id = traci.vehicle.getRoadID(veh_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                total_waiting_time += wait_time_car
        return total_waiting_time
    # RETRIEVE THE WAITING TIME OF EVERY CAR IN THE INCOMING LANES
    
    def get_intersection_q_per_step(self):
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        intersection_queue = halt_N + halt_S + halt_E + halt_W
        return intersection_queue

    def get_stops_per_step(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        stops = sum(traci.edge.getLastStepHaltingNumber(r) for r in incoming_roads)
        return stops

    def get_vehicle_count_per_step(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        count = sum(traci.edge.getLastStepVehicleNumber(r) for r in incoming_roads)
        return count

    def get_average_speed_per_step(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        speeds = [traci.edge.getLastStepMeanSpeed(r) for r in incoming_roads]
        # filter out edges with no vehicles (SUMO returns -1 or 0 for empty edges)
        active = [s for s in speeds if s > 0]
        return sum(active) / len(active) if active else 0.0

    def get_co2_per_step(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        co2 = sum(traci.edge.getCO2Emission(r) for r in incoming_roads)
        return co2

    def _encode_env_state( self ):
        state = np.zeros(80)  # binary position cells only; extra features appended below

        for veh_id in traci.vehicle.getIDList():
            lane_pos = traci.vehicle.getLanePosition(veh_id)
            lane_id = traci.vehicle.getLaneID(veh_id)
            lane_pos = self.SUMO_INT_LANE_LENGTH - lane_pos  # inversion of lane pos, so if the car is close to TL, lane_pos = 0
            lane_group = -1  # just dummy initialization
            is_car_valid = False  # flag for not detecting cars crossing the intersection or driving away from it

            # distance in meters from the TLS -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 200:
                lane_cell = 7
            elif lane_pos < 350:
                lane_cell = 8
            elif lane_pos <= 500:
                lane_cell = 9

            # Isolate the  "turn left only" from "straight" and "right" turning lanes.
            # This is because TL lights are turned on separately for these sets
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7

            if lane_group >= 1 and lane_group <= 7:
                veh_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                is_car_valid = True
            elif lane_group == 0:
                veh_position = lane_cell
                is_car_valid = True

            if is_car_valid:
                state[veh_position] = 1  # write the position of the car veh_id in the state array

        extra = np.array([
            self.get_intersection_q_per_step() / 500,
            self.get_vehicle_count_per_step() / 500,
            self.get_average_speed_per_step() / 25,
            traci.trafficlight.getPhase("TL") / 7,
            self.get_stops_per_step() / 500,
            self.get_co2_per_step() / 100000,
            self.steps / self.max_steps,
            traci.simulation.getMinExpectedNumber() / 500,
        ])
        return np.concatenate([state, extra])
        
    def __del__( self ):
        try:
            traci.close()
        except Exception:
            # Ignore errors when TRACI is already disconnected
            pass
