import dgl
import torch
import numpy as np
from dgl import DGLGraph
from dgl.transforms import BaseTransform


class InductiveGraphSplit(BaseTransform):

    def __init__(self, num_test:float,
                       num_val :float):
        
        self.train_split = 1-num_test-num_val
        self.test_split = num_test
        self.valid_split = num_val  
    
    def __call__(self, graph:DGLGraph)->DGLGraph:
        num_partitions = 10
        num_train = self.train_split*num_partitions
        num_test = self.test_split*num_partitions
        num_valid =self.valid_split*num_partitions
        while ((num_train != int(num_train)) and \
               (num_test  != int(num_test)) and \
               (num_valid != int(num_valid))):
            num_partitions *= num_partitions
            num_train = self.train_split*num_partitions
            num_test = self.test_split*num_partitions
            num_valid =self.valid_split*num_partitions

        num_train = int(num_train)
        num_valid = int(num_valid)
        num_test = int(num_test)

        partitions = dgl.metis_partition(graph, num_partitions)
        train_node_list = list()
        test_node_list = list()
        valid_node_list = list()

        key = 0
        for _ in range(num_train):
            train_node_list.append(partitions[key].ndata["_ID"])
            key+=1
        for _ in range(num_valid):
            valid_node_list.append(partitions[key].ndata["_ID"])
            key+=1
        for _ in range(num_test):
            test_node_list.append(partitions[key].ndata["_ID"])
            
        print(len(train_node_list), len(valid_node_list), len(test_node_list))

        # Find unnecessary edges
        src, dst = graph.edges()
        senders = src.numpy()
        receivers = dst.numpy()

        try:
            train_nodes = np.hstack(train_node_list)
        except:
            train_nodes = np.array(train_nodes)
        try:
            valid_nodes = np.hstack(valid_node_list)
        except:
            valid_nodes = np.array(valid_node_list)
        try:
            test_nodes = np.hstack(test_node_list)
        except:
            test_nodes = np.array(test_nodes)

        # remove unnecessary edges
        train_sender_edge_mask = np.isin(senders, train_nodes)
        valid_sender_edge_mask = np.isin(senders, valid_nodes)
        test_sender_edge_mask = np.isin(senders, test_nodes)

        train_receiver_edge_mask = np.isin(receivers, train_nodes)
        valid_receiver_edge_mask = np.isin(receivers, valid_nodes)
        test_receiver_edge_mask = np.isin(receivers, test_nodes)

        # Get unnecesary edge masks
        train_test_edge_mask = np.logical_and(train_sender_edge_mask, test_receiver_edge_mask)
        train_valid_edge_mask = np.logical_and(train_sender_edge_mask, valid_receiver_edge_mask)
        valid_train_edge_mask = np.logical_and(valid_sender_edge_mask, train_receiver_edge_mask)
        valid_test_edge_maks = np.logical_and(valid_sender_edge_mask, test_receiver_edge_mask)
        test_train_edge_mask = np.logical_and(test_sender_edge_mask, train_receiver_edge_mask)
        test_valid_edge_mask = np.logical_and(test_sender_edge_mask, valid_receiver_edge_mask)

        # Total removing edges
        removing_edges_mask = np.logical_or(train_test_edge_mask,
                                    np.logical_or(train_valid_edge_mask,
                                        np.logical_or(valid_train_edge_mask,
                                            np.logical_or(valid_test_edge_maks,
                                                np.logical_or(test_train_edge_mask,
                                                              test_valid_edge_mask)))))
        removing_edges_idx = np.where(removing_edges_mask == True)[0]

        removing_eids = graph.edge_ids(src[removing_edges_idx], dst[removing_edges_idx])
        
        graph.remove_edges(removing_eids)
        print("remove edges for inductive setting")
        print("Number of removed edges: ", len(removing_eids))

        # Set train/valid/test mask
        total_nodes = np.arange(graph.num_nodes())
        train_mask = np.isin(total_nodes, train_nodes)
        valid_mask = np.isin(total_nodes, valid_nodes)
        test_mask = np.isin(total_nodes, test_nodes)

        graph.ndata["train_mask"] = torch.from_numpy(train_mask)
        graph.ndata["test_mask"] = torch.from_numpy(test_mask)
        graph.ndata["valid_mask"] = torch.from_numpy(valid_mask)

        return graph


    def depreicated__call__(self, graph:DGLGraph)->DGLGraph:
        num_partitions = 10
        num_train = self.train_split*num_partitions
        num_test = self.test_split*num_partitions
        num_valid =self.valid_split*num_partitions
        while ((num_train != int(num_train)) and \
               (num_test  != int(num_test)) and \
               (num_valid != int(num_valid))):
            num_partitions *= num_partitions
            num_train = self.train_split*num_partitions
            num_test = self.test_split*num_partitions
            num_valid =self.valid_split*num_partitions

        num_train = int(num_train)
        num_valid = int(num_valid)
        num_test = int(num_test)

        partitions = dgl.metis_partition(graph, num_partitions)
        train_node_list = list()
        test_node_list = list()
        valid_node_list = list()

        key = 0
        for _ in range(num_train):
            train_node_list.append(partitions[key].ndata["_ID"])
            key+=1
        for _ in range(num_valid):
            valid_node_list.append(partitions[key].ndata["_ID"])
            key+=1
        for _ in range(num_test):
            test_node_list.append(partitions[key].ndata["_ID"])

        # Find unnecessary edges
        src, dst = graph.edges()
        senders = src.numpy()
        receivers = dst.numpy()

        try:
            train_nodes = np.hstack(train_node_list)
        except:
            train_nodes = np.array(train_nodes)
        try:
            valid_nodes = np.hstack(valid_node_list)
        except:
            valid_nodes = np.array(valid_node_list)
        try:
            test_nodes = np.hstack(test_node_list)
        except:
            test_nodes = np.array(test_nodes)

        # remove unnecessary edges
        train_sender_edge_mask = np.isin(senders, train_nodes)
        valid_sender_edge_mask = np.isin(senders, valid_nodes)
        test_sender_edge_mask = np.isin(senders, test_nodes)

        train_receiver_edge_mask = np.isin(receivers, train_nodes)
        valid_receiver_edge_mask = np.isin(receivers, valid_nodes)
        test_receiver_edge_mask = np.isin(receivers, test_nodes)

        # Get unnecesary edge masks
        train_test_edge_mask = np.logical_and(train_sender_edge_mask, test_receiver_edge_mask)
        train_valid_edge_mask = np.logical_and(train_sender_edge_mask, valid_receiver_edge_mask)
        valid_train_edge_mask = np.logical_and(valid_sender_edge_mask, train_receiver_edge_mask)
        valid_test_edge_maks = np.logical_and(valid_sender_edge_mask, test_receiver_edge_mask)
        test_train_edge_mask = np.logical_and(test_sender_edge_mask, train_receiver_edge_mask)
        test_valid_edge_mask = np.logical_and(test_sender_edge_mask, valid_receiver_edge_mask)

        # Total removing edges
        removing_edges_mask = np.logical_or(train_test_edge_mask,
                                    np.logical_or(train_valid_edge_mask,
                                        np.logical_or(valid_train_edge_mask,
                                            np.logical_or(valid_test_edge_maks,
                                                np.logical_or(test_train_edge_mask,
                                                              test_valid_edge_mask)))))
        removing_edges_idx = np.where(removing_edges_mask == True)[0]

        graph.remove_edges(removing_edges_idx)
        print("remove edges for inductive setting")

        # Set train/valid/test mask
        total_nodes = np.arange(graph.num_nodes())
        train_mask = np.isin(total_nodes, train_nodes)
        valid_mask = np.isin(total_nodes, valid_nodes)
        test_mask = np.isin(total_nodes, test_nodes)

        graph.ndata["train_mask"] = torch.from_numpy(train_mask)
        graph.ndata["test_mask"] = torch.from_numpy(test_mask)
        graph.ndata["valid_mask"] = torch.from_numpy(valid_mask)

        return graph


        
            

        