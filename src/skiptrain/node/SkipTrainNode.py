import importlib
import json
import logging
import math
import os
from collections import deque
import pandas as pd
import numpy as np


from decentralizepy.node.DPSGDNode import DPSGDNode
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping


class SkipTrainNode(DPSGDNode):
    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        *args
    ):
        """
        Construct objects.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations.
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """
        logging.info("Started process.")

        self.init_log(log_dir, rank, log_level)

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            graph,
            iterations,
            log_dir,
            weights_store_dir,
            test_after,
            train_evaluate_after,
            reset_optimizer,
        )
        self.init_dataset_model(config["DATASET"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])
        self.init_comm(config["COMMUNICATION"])
        self.init_node(config["NODE"])
        self.random_generator = np.random.default_rng(1000*rank + config["DATASET"]["random_seed"])

        self.message_queue = dict()

        self.barrier = set()
        self.my_neighbors = self.graph.neighbors(self.uid)

        self.init_sharing(config["SHARING"])
        self.peer_deques = dict()
        self.connect_neighbors()
    
    def init_node(self, node_configs):
        self.remaining_rounds = None
        self.prob_policy = node_configs["algorithm"]
        energy_traces_path = node_configs["energy_traces"]
        energy_traces = pd.read_csv(energy_traces_path)
        
        if node_configs["algorithm"] == "skiptrain":
            self.consumption = energy_traces.iloc[self.rank % energy_traces.shape[0]]["Energy (Wh)"]
            self.consecutive_training_rounds = node_configs["consecutive_training_rounds"]
            self.consecutive_synching_rounds = node_configs["consecutive_synching_rounds"]
        elif node_configs["algorithm"] == "skiptrain_constrained":
            self.consumption = energy_traces.iloc[self.rank % energy_traces.shape[0]]["Energy (Wh)"]
            self.remaining_rounds = energy_traces.iloc[self.rank % energy_traces.shape[0]]["Number of training rounds"]
            self.consecutive_training_rounds = node_configs["consecutive_training_rounds"]
            self.consecutive_synching_rounds = node_configs["consecutive_synching_rounds"]
            max_training_rounds = (self.consecutive_training_rounds/(self.consecutive_training_rounds +  self.consecutive_synching_rounds))*float(self.iterations)
            self.training_prob = min(self.remaining_rounds/max_training_rounds,1)
        elif node_configs["algorithm"] == "dpsgd": 
            self.consumption = energy_traces.iloc[self.rank % energy_traces.shape[0]]["Energy (Wh)"]
        elif node_configs["algorithm"] == "greedy":
            self.consumption = energy_traces.iloc[self.rank % energy_traces.shape[0]]["Energy (Wh)"]
            self.remaining_rounds = energy_traces.iloc[self.rank % energy_traces.shape[0]]["Number of training rounds"]

        self.averaging = node_configs["averaging"]

    def run(self):
        """
        Start the decentralized learning

        """
        rounds_to_test = self.test_after
        rounds_to_train_evaluate = self.train_evaluate_after
        global_epoch = 1
        change = 1

        for iteration in range(self.iterations):
            logging.info("Starting training iteration: %d", iteration)
            rounds_to_train_evaluate -= 1
            rounds_to_test -= 1

            self.iteration = iteration

            if self.prob_policy == "skiptrain":
                if iteration % (self.consecutive_training_rounds + self.consecutive_synching_rounds) < self.consecutive_training_rounds:
                    logging.info("Training round")
                    self.prob = 1
                else:
                    logging.info("Syncronization round")
                    self.prob = 0
            elif self.prob_policy == "skiptrain_constrained":
                 if (iteration % (self.consecutive_training_rounds + self.consecutive_synching_rounds) < self.consecutive_training_rounds) and self.remaining_rounds > 0:
                    logging.info("Training round")
                    self.prob = self.training_prob
                 else:
                    logging.info("Syncronization round")
                    self.prob = 0
            elif self.prob_policy == "dpsgd":
                self.prob = 1.0
            elif self.prob_policy == "greedy":
                if self.remaining_rounds > 0:
                    self.prob = 1.0
                else:
                    self.prob = 0
            action = self.random_generator.choice(["p", "np"], p=[self.prob, 1-self.prob])
            logging.info(f"Action: {action}")

            # Training
            if action == "p":
                logging.info("Training")
                if self.prob_policy == "skiptrain_constrained" or self.prob_policy == "greedy":
                    self.remaining_rounds -= 1
                self.trainer.train(self.dataset)
            else:
                logging.info("Not training this round")

            # Sharing
            new_neighbors = self.get_neighbors()

            self.my_neighbors = new_neighbors
            self.connect_neighbors()
            logging.debug("Connected to all neighbors")

            to_send = self.sharing.get_data_to_send(degree=len(self.my_neighbors))
            to_send["CHANNEL"] = "DPSGD"

            for neighbor in self.my_neighbors:
                logging.info(f"Sending to {neighbor}")
                logging.debug(f"Data to be sent: {to_send}")
                self.communication.send(neighbor, to_send)

            while not self.received_from_all():
                sender, data = self.receive_DPSGD()
                logging.debug(
                    "Received Model from {} of iteration {}".format(
                        sender, data["iteration"]
                    )
                ) 
                logging.debug(f"Received data: {data}") 
                if sender not in self.peer_deques:
                    self.peer_deques[sender] = deque()

                if data["iteration"] == self.iteration:
                    self.peer_deques[sender].appendleft(data)
                else:
                    self.peer_deques[sender].append(data)
                logging.debug("Peer dequeue")
                logging.debug(self.peer_deques)
            logging.info("Received from all concluded")

            # Aggregation
            averaging_deque = dict()
            for neighbor in self.my_neighbors:
                logging.info(f"Added data from node {neighbor} to the averaging deque")
                averaging_deque[neighbor] = self.peer_deques[neighbor]

            if self.averaging == "Metro-Hasting":
                logging.info("Metro-Hasting average")
                self.sharing._averaging(averaging_deque)
            else:
                logging.info("Weighted average")
                raise NotImplementedError

            model_params = self.sharing.model.state_dict()
            logging.debug(f"Averaged model: {model_params['conv1.weight'][0][0][0]}")
            if self.reset_optimizer:
                self.optimizer = self.optimizer_class(
                    self.model.parameters(), **self.optimizer_params
                )  # Reset optimizer state
                self.trainer.reset_optimizer(self.optimizer)

            # Data logging
            if iteration:
                with open(
                    os.path.join(self.log_dir, "{}_results.json".format(self.rank)),
                    "r",
                ) as inf:
                    results_dict = json.load(inf)
            else:
                results_dict = {
                    "train_loss": {},
                    "test_loss": {},
                    "test_acc": {},
                    "validation_loss": {},
                    "validation_acc": {},
                    "total_bytes": {},
                    "total_meta": {},
                    "total_data_per_n": {},
                    "training_energy_traces": {},
                    "action": {},
                    "training_prob": {},
                    "remaining_training_rounds": {}
                }
                results_dict["training_energy_traces"]["0"] = float(self.consumption)

            if self.prob_policy=="skiptrain_constrained" or self.prob_policy=="greedy":
                results_dict["remaining_training_rounds"][iteration + 1] = int(self.remaining_rounds)
            results_dict["training_prob"][iteration + 1] = float(self.prob)
            results_dict["total_bytes"][iteration + 1] = self.communication.total_bytes
            results_dict["action"][iteration + 1] = action
        
            if hasattr(self.communication, "total_meta"):
                results_dict["total_meta"][
                    iteration + 1
                ] = self.communication.total_meta
            if hasattr(self.communication, "total_data"):
                results_dict["total_data_per_n"][
                    iteration + 1
                ] = self.communication.total_data

            if rounds_to_train_evaluate == 0:
                logging.info("Evaluating on train set.")
                rounds_to_train_evaluate = self.train_evaluate_after * change
                loss_after_sharing = self.trainer.eval_loss(self.dataset)
                results_dict["train_loss"][iteration + 1] = loss_after_sharing

            if self.dataset.__testing__ and rounds_to_test == 0:
                rounds_to_test = self.test_after * change
                logging.info("Evaluating on test set.")
                ta, tl = self.dataset.test(self.model, self.loss)
                results_dict["test_acc"][iteration + 1] = ta
                results_dict["test_loss"][iteration + 1] = tl
                if self.dataset.__validating__:
                    logging.info("Evaluating on the validation set")
                    va, vl = self.dataset.validate(self.model, self.loss)
                    results_dict["validation_acc"][iteration + 1] = va
                    results_dict["validation_loss"][iteration + 1] = vl


                if global_epoch == 49:
                    change *= 1

                global_epoch += change

            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
            ) as of:
                json.dump(results_dict, of)
        if self.model.shared_parameters_counter is not None:
            logging.info("Saving the shared parameter counts")
            with open(
                os.path.join(
                    self.log_dir, "{}_shared_parameters.json".format(self.rank)
                ),
                "w",
            ) as of:
                json.dump(self.model.shared_parameters_counter.numpy().tolist(), of)
        self.disconnect_neighbors()
        logging.info("Storing final weight")
        self.model.dump_weights(self.weights_store_dir, self.uid, iteration)
        logging.info("All neighbors disconnected. Process complete!")