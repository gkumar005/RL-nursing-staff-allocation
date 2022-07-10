#!/usr/bin/env python
# coding: utf-8

# # MY ED Simulation

# In[118]:


import simpy
import numpy as np
import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt


# In[119]:


class g:
    cls_0ar = 50
    cls_1ar = 20
    cls_2ar = 15
    cls_3ar = 20
    cls_4ar = 50
    
    sr_reg_triage = 10 # reception has equal service time for each patients
    sr_rescue = 70 # mean service time at node = 2, resucitation center for class 0
    sr_clas_1_2 = 30 # mean service time at node = 3, (class - II & class III)
    sr_clas_3_4 = 20 # mean service time at node = 4, (class - IV & class V)
    
    number_of_reg_triage_nurses = 2
    number_of_rescue_nurses = 3
    number_of_nurses_node_3 = 3
    number_of_nurses_node_4 = 2
    
    sim_duration = 2880
    warm_up_duration = 1440
    number_of_runs = 1


# In[120]:


class ED_Patient_clas_0:
    def __init__(self, p_id):
        self.id = p_id
        self.priority = 0
        self.start_q_rescue = 0
        self.q_time_rescue = 0
        self.sampled_rescue_duration = 0
        
class ED_Patient_clas_1:
    def __init__(self, p_id):
        self.id = p_id
        self.priority = 1
        self.start_q_reg_triage=0
        self.q_time_reg_triage = 0
        self.sampled_reg_triage_duration = 0
        self.q_time_consultation_node_3 = 0
        
class ED_Patient_clas_2:
    def __init__(self, p_id):
        self.id = p_id
        self.priority = 2
        self.start_q_reg_triage=0
        self.q_time_reg_triage = 0
        self.sampled_reg_triage_duration = 0
        self.q_time_consultation_node_3 = 0

class ED_Patient_clas_3:
    def __init__(self, p_id):
        self.id = p_id
        self.priority = 3
        self.start_q_reg_triage=0
        self.q_time_reg_triage = 0
        self.sampled_reg_triage_duration = 0
        self.q_time_consultation_node_4 = 0
        
class ED_Patient_clas_4:
    def __init__(self, p_id):
        self.id = p_id
        self.priority = 4
        self.start_q_reg_triage=0
        self.q_time_reg_triage = 0
        self.sampled_reg_triage_duration = 0
        self.q_time_consultation_node_4 = 0


# In[132]:



class SIMULATION_ED:
    
    def __init__(self, render_env=False, sim_duration=24*60, time_step=120):
        # simulation results
        self.result_df = pd.DataFrame()
        self.result_df['PID'] = []
        self.result_df['class0_ar'] = []
        self.result_df['Q_Time_Rescue'] = []
        self.result_df['class0_sr'] = []
        self.result_df['other_class_ar'] = []
        self.result_df['Q_Time_Reg_Traige'] = []
        self.result_df['other_class_reg_sr'] = []
        self.result_df['Q_Time_Consultation_n3'] = []
        self.result_df['Q_Time_Consultation_n4'] = []
        self.result_df['Priority'] = []
        
        self.result_df.set_index("PID", inplace = True)
        
        ## STATES of MODEL##
        
        # Set up state dictionary 
        self.state = dict()
        
        # Q length at Node-1(registration and Traige)
        self.state['QN1'] = 0
        # Q length at Node-2(rescusitation centre)
        self.state['QN2'] = 0
        # Q length at Node-3(Consultation centre for class - 1&2)
        self.state['QN3'] = 0
        # Q length at Node-4(Consultation centre for class - 3&4)
        self.state['QN4'] = 0
        
        # nurses at Node-1(registration and Traige)
        self.state['Nurses_N1'] = 0
        # nurses at Node-2(rescusitation centre)
        self.state['Nurses_N2'] = 0
        # nurses at Node-3(Consultation centre for class - 1&2)
        self.state['Nurses_N3'] = 0
        # nurses at Node-4(Consultation centre for class - 3&4)
        self.state['Nurses_N4'] = 0
        
        # Show environemnt on each action?
        self.render_env = render_env
        
        ## ACTIONS of MODEL##
        
        # Add K,b as labels to the graph to make things simple for other functions
        # K = number of nodes b = number of nurses
        K = 4  # self.tot_nodes
        b = 10 # self.tot_nurses
        G = nx.DiGraph(K=4,b=10)
        s = (0,0)
        d= (K,b)

        # Nodes are indexed (i,j) in the way described by the paper
        # Weights are initialised to 1
        # Add source node
        G.add_node(s)

        # Add nodes and directed edges from source to first layer of graph
        for j in range(b+1):
            G.add_node((1,j))
            G.add_weighted_edges_from([(s,(1,j),1)])

        # Add nodes for each layer and directed edges
        for i in range(2,K):
            for j in range(b+1):
                G.add_node((i,j))
                G.add_weighted_edges_from([((i-1,y),(i,j),1) for y in range(j+1)])

        # Add destination node
        G.add_node(d)
        
            # Add directed edges to destination node
        for j in range(b+1):
            G.add_weighted_edges_from([((K-1,j),d,1)])
        
        all_paths = nx.all_simple_paths(G, source = (0,0), target=(K,b), cutoff=None)
        all_paths = list(all_paths)
        self.action_space = []
        for i in list(all_paths):
            n1 = i[1][1]-i[0][1]
            n2 = i[2][1]-i[1][1]
            n3 = i[3][1]-i[2][1]
            n4 = i[4][1]-i[3][1]
            if n1>0 and n2>0 and n3>0 and n4>0:
                self.action_space.append([n1,n2,n3,n4])
        
        # action index
        self.actions = list(range(len(self.action_space)))
        
        
        # Set sim duration (returns Terminal state after this) and time steps
        self.sim_duration = sim_duration
        self.time_step = time_step
        self.next_time_stop = 0
        
        # Set up observation and action space sizes
        self.observation_size = 8
        self.action_size = len(self.action_space)    
            
    
    def _calculate_reward(self):
        """
        Calculate reward (always negative or 0)
        """
        
        wait_time_cl0 = np.mean(self.result_df[self.result_df.Priority==0].Q_Time_Rescue)
        
        wait_time_cl1 = np.mean(np.array(self.result_df[self.result_df.Priority==1].Q_Time_Reg_Traige)+
                                np.array(self.result_df[self.result_df.Priority==1].Q_Time_Consultation_n3))
        wait_time_cl2 = np.mean(np.array(self.result_df[self.result_df.Priority==2].Q_Time_Reg_Traige)+
                                np.array(self.result_df[self.result_df.Priority==2].Q_Time_Consultation_n3))
        wait_time_cl3 = np.mean(np.array(self.result_df[self.result_df.Priority==3].Q_Time_Reg_Traige)+
                                np.array(self.result_df[self.result_df.Priority==3].Q_Time_Consultation_n4))
        wait_time_cl4 = np.mean(np.array(self.result_df[self.result_df.Priority==4].Q_Time_Reg_Traige)+
                                np.array(self.result_df[self.result_df.Priority==4].Q_Time_Consultation_n4))
        
        wait_t_ls = [wait_time_cl0,wait_time_cl1,wait_time_cl2,wait_time_cl3,wait_time_cl4]
        for i,val in enumerate(wait_t_ls):
            if str(val) == 'nan':
                wait_t_ls[i] = 0
                
                
        
        # loss = negative value of diffrence in spare beds from target spare beds
        loss = -abs(wait_t_ls[0]*20 + wait_t_ls[1]*1 + wait_t_ls[2]*1 + wait_t_ls[3]*0.5 + wait_t_ls[4]*0.25)
                    
        return loss
    
    
    def _get_observations(self):
        """Returns current state observation"""
        self.state['QN1'] = len(self.reg_triage_nurse.queue)
        self.state['QN2'] = len(self.rescue_nurse.queue)
        self.state['QN3'] = len(self.node_3_consultation_nurse.queue)
        self.state['QN4'] = len(self.node_4_consultation_nurse.queue)
        self.state['Nurses_N1'] = g.number_of_reg_triage_nurses
        self.state['Nurses_N2'] = g.number_of_rescue_nurses
        self.state['Nurses_N3'] = g.number_of_nurses_node_3
        self.state['Nurses_N4'] = g.number_of_nurses_node_4
        
        # Put state dictionary items into observations list
        observations = [v for k,v in self.state.items()]  
        # Return starting state observations
        return observations
    
    
    def _islegal(self, action):
        """
        Check action is in list of allowed actions. If not, raise an exception.
        """
        
        if action not in self.actions:
            raise ValueError('Requested action not in list of allowed actions') 
     
    
    def render(self):
        """Display current state"""
        
        print (self.state)
        
    
    def reset(self):
        """Reset environemnt"""
        
        # Initialise simpy environemnt
        # Parameter Initiallisation
        self.env = simpy.Environment()
        self.patient_counter = 0
        self.reg_triage_nurse = simpy.Resource(self.env, capacity = g.number_of_reg_triage_nurses)
        self.rescue_nurse = simpy.Resource(self.env, capacity = g.number_of_rescue_nurses)
        self.node_3_consultation_nurse = simpy.PriorityResource(self.env, capacity = g.number_of_nurses_node_3)
        self.node_4_consultation_nurse = simpy.PriorityResource(self.env, capacity = g.number_of_nurses_node_4)
        
        self.result_df = pd.DataFrame()
        self.result_df['PID'] = []
        self.result_df['class0_ar'] = []
        self.result_df['Q_Time_Rescue'] = []
        self.result_df['class0_sr'] = []
        self.result_df['other_class_ar'] = []
        self.result_df['Q_Time_Reg_Traige'] = []
        self.result_df['other_class_reg_sr'] = []
        self.result_df['Q_Time_Consultation_n3'] = []
        self.result_df['Q_Time_Consultation_n4'] = []
        self.result_df['Priority'] = []
        self.result_df.set_index("PID", inplace = True)
        
        self.next_time_stop = 0

        # Set starting state values
        self.state['QN1'] = 0
        self.state['QN2'] = 0
        self.state['QN3'] = 0
        self.state['QN4'] = 0
        self.state['Nurses_N1'] = 0
        self.state['Nurses_N2'] = 0
        self.state['Nurses_N3'] = 0
        self.state['Nurses_N4'] = 0
        
        # Return starting state observations
        observations = self._get_observations()
        return observations
        
    def generate_cls_0_arrivals(self):
        while True:
            self.patient_counter += 1
            P_cl0 = ED_Patient_clas_0(self.patient_counter)

            self.env.process(self.cls_0_patient_rescue_journey(P_cl0))

            sampled_cls_0_interarrival = random.expovariate(1.0/g.cls_0ar)

            yield self.env.timeout(sampled_cls_0_interarrival)
    
    def generate_cls_1_arrivals(self):
        while True:
            self.patient_counter += 1
            P_cl1 = ED_Patient_clas_1(self.patient_counter)

            self.env.process(self.other_patient_journey(P_cl1))

            sampled_cls_1_interarrival = random.expovariate(1.0/g.cls_1ar)

            yield self.env.timeout(sampled_cls_1_interarrival)
            
    def generate_cls_2_arrivals(self):
        while True:
            self.patient_counter += 1
            P_cl2 = ED_Patient_clas_2(self.patient_counter)

            self.env.process(self.other_patient_journey(P_cl2))

            sampled_cls_2_interarrival = random.expovariate(1.0/g.cls_2ar)

            yield self.env.timeout(sampled_cls_2_interarrival)
            
    def generate_cls_3_arrivals(self):
        while True:
            self.patient_counter += 1
            P_cl3 = ED_Patient_clas_3(self.patient_counter)

            self.env.process(self.other_patient_journey(P_cl3))

            sampled_cls_3_interarrival = random.expovariate(1.0/g.cls_3ar)

            yield self.env.timeout(sampled_cls_3_interarrival)
            
    def generate_cls_4_arrivals(self):
        while True:
            self.patient_counter += 1
            P_cl4 = ED_Patient_clas_4(self.patient_counter)

            self.env.process(self.other_patient_journey(P_cl4))

            sampled_cls_4_interarrival = random.expovariate(1.0/g.cls_4ar)

            yield self.env.timeout(sampled_cls_4_interarrival)
                
    def cls_0_patient_rescue_journey(self,patient):
        ###Registration
        patient.start_q_rescue = self.env.now
        with self.rescue_nurse.request() as req:
            yield req

            end_q_rescue = self.env.now

            patient.q_time_rescue = end_q_rescue - patient.start_q_rescue
            
            patient.sampled_rescue_duration = random.expovariate(1.0/g.sr_rescue)

            yield self.env.timeout(patient.sampled_rescue_duration)
    
        self.store_patient_results(patient)
    
    def other_patient_journey(self,patient):
        ###Registration
        patient.start_q_reg_triage = self.env.now
        #print("PID",patient.id,"arrival time", start_q_reg_triage)
        with self.reg_triage_nurse.request() as req:
            yield req
            
            end_q_reg_triage = self.env.now
            
            patient.q_time_reg_triage = end_q_reg_triage - patient.start_q_reg_triage

            patient.sampled_reg_triage_duration = random.expovariate(1.0/g.sr_reg_triage)
            #print("PID",patient.id,"  time at reg", end_q_reg_triage, " sr time", sampled_reg_triage_duration)
            yield self.env.timeout(patient.sampled_reg_triage_duration)
            
        if patient.priority in [1,2]:
            start_q_node_3 = self.env.now
            with self.node_3_consultation_nurse.request(priority = patient.priority) as req:
                yield req

                end_q_node_3 = self.env.now
                #print("PID",patient.id,"  time at Node3", end_q_node_3)
                patient.q_time_consultation_node_3 = end_q_node_3 - start_q_node_3

                sampled_consultation_node_3_duration = random.expovariate(1.0/g.sr_clas_1_2)

                yield self.env.timeout(sampled_consultation_node_3_duration)
                
        else:
            start_q_node_4 = self.env.now
             
            with self.node_4_consultation_nurse.request(priority = patient.priority) as req:
                yield req

                end_q_node_4 = self.env.now

                patient.q_time_consultation_node_4 = end_q_node_4 - start_q_node_4
                
                sampled_consultation_node_4_duration = random.expovariate(1.0/g.sr_clas_3_4)

                yield self.env.timeout(sampled_consultation_node_4_duration)          
                

        self.store_patient_results(patient)

    def store_patient_results(self,patient):
        #print(1)
        if patient.priority == 0:
            patient.q_time_reg_triage = float('nan')
            patient.q_time_consultation_node_3 = float('nan')
            patient.q_time_consultation_node_4 = float('nan')
            patient.start_q_reg_triage = float('nan')
            patient.sampled_reg_triage_duration = float('nan')
        elif patient.priority in [1,2]:
            patient.q_time_rescue = float('nan')
            patient.start_q_rescue = float('nan')
            patient.sampled_rescue_duration = float('nan')
            patient.q_time_consultation_node_4 = float('nan')
        else:
            patient.q_time_rescue = float('nan')
            patient.start_q_rescue = float('nan')
            patient.sampled_rescue_duration = float('nan')
            patient.q_time_consultation_node_3 = float('nan')
        
        df_to_add = pd.DataFrame({"PID":[patient.id],
                                  "class0_ar":[patient.start_q_rescue],
                                  "Q_Time_Rescue":[patient.q_time_rescue],
                                  "class0_sr":[patient.sampled_rescue_duration],
                                  "other_class_ar":[patient.start_q_reg_triage],
                                  "Q_Time_Reg_Traige":[patient.q_time_reg_triage],
                                  "other_class_sr":[patient.sampled_reg_triage_duration],
                                  "Q_Time_Consultation_n3":[patient.q_time_consultation_node_3],
                                  "Q_Time_Consultation_n4":[patient.q_time_consultation_node_4],
                                  "Priority":[patient.priority]})
        df_to_add.set_index("PID", inplace = True)
        self.result_df =  self.result_df.append(df_to_add)


    def step(self, action, ctr = 1):
        
        # Check action is legal (raise exception if not):
        self._islegal(action)
        
        g.number_of_reg_triage_nurses = self.action_space[action][0]
        g.number_of_rescue_nurses = self.action_space[action][1]
        g.number_of_nurses_node_3 = self.action_space[action][2]
        g.number_of_nurses_node_4 = self.action_space[action][3]
        
        if ctr == 1:
            self.env.process(self.generate_cls_0_arrivals())
            self.env.process(self.generate_cls_1_arrivals())
            self.env.process(self.generate_cls_2_arrivals())
            self.env.process(self.generate_cls_3_arrivals())
            self.env.process(self.generate_cls_4_arrivals())
        
        # Make a step in the simulation
        self.next_time_stop += self.time_step
        self.env.run(until=self.next_time_stop)
        
        # Get new observations
        observations = self._get_observations()
        
        # Check whether terminal state reached (based on sim time)
        terminal = True if self.env.now >= self.sim_duration else False
        
        # Get reward
        reward = self._calculate_reward()
        
        # Information is empty dictionary (used to be compatble with OpenAI Gym)
        info = dict()
        
        # Render environment if requested
        if self.render_env:
            self.render()
        
        # Return tuple of observations, reward, terminal, info
        return (observations, reward, terminal, info)


# In[133]:


# #random.seed(0)
# msimul = SIMULATION_ED(render_env=False, sim_duration=24*60, time_step=120)


# In[134]:


# msimul.reset()
# ctr = 1
# for i in range(12):
#     b = random.randint(1,83)
#     #print(b)
#     print(msimul.step(b,ctr))
#     ctr += 1


# In[135]:


# (sorted(msimul.result_df.index.unique()))


# In[136]:


# msimul.result_df[msimul.result_df.Priority == 0]


# In[137]:


# msimul.reg_triage_nurse.count+msimul.rescue_nurse.count+msimul.node_3_consultation_nurse.count+msimul.node_4_consultation_nurse.count


# In[138]:


# len(msimul.reg_triage_nurse.queue+msimul.rescue_nurse.queue+msimul.node_3_consultation_nurse.queue+msimul.node_4_consultation_nurse.queue)


# In[139]:


# str(msimul.result_df.Q_Time_Reg_Traige[1])


# In[ ]:


# a = []
# for i in action:
#     a += [list(np.array([-1,0,1])+i)]
# a
# import itertools
# action_combination = list(itertools.product(*a))

# k = 0
# for i in action_combination:
#     if sum(i) == 15 and (i[0] >=0 and i[1] >=0 and i[2] >=0 and i[3]>=0):
#         print(i)
#         k+=1
# k


# In[ ]:





# In[ ]:




