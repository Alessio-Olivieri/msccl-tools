import logging
from dataclasses import dataclass
from enum import Enum
from msccl.language.ir import *
from msccl.language.passes import *
from msccl.language.tb_assignment import *
from msccl.language.chunk import *
from msccl.language.buffer import *
from msccl.language.rank_dag import *
import msccl.collectives as collectives

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

_current_program = None

def _curr():
    global _current_program
    if _current_program == None:
        raise RuntimeError("No Program in context")
    return _current_program

class MSCCLProgram:
    def __init__(self, name, topo, collective, instances, protocol='Simple', \
            threadblock_policy=ThreadblockPolicy.auto, interleaved_replication=True,
            instr_fusion=True, check_xml=True, dependence_nop=False):
        logging.info(f"Initializing MSCCLProgram with name: {name}, protocol: {protocol}")
        self.name = name
        self.topo = topo
        self.collective = collective       
        self.num_ranks = topo.num_nodes()
        self.instances = instances
        self.protocol = protocol
        self.threadblock_policy = threadblock_policy
        self.interleaved_replication = interleaved_replication
        self.instr_fusion = instr_fusion
        self.check_xml = check_xml
        self.dependence_nop = dependence_nop
        assert protocol in ['Simple', 'LL', 'LL128'], \
            f'Given protocol: {protocol}. Must be either Simple, LL, LL128'
        self.run_opt = True # Runs optimization passes
        self.buffers = collective.init_buffers()
        self.instr_dag = InstructionDAG(self.num_ranks, self.buffers)
        
        logging.debug(f"MSCCLProgram initialized with {self.num_ranks} ranks.")
        
        for r in range(self.num_ranks):
            logging.debug(f"Initializing buffers for rank {r}")
            for index, chunk in enumerate(self.buffers[r][Buffer.input]):
                buffer, index = self.collective.get_buffer_index(r, Buffer.input, index)
                ref = self.get_ref(r, buffer, index, 1)
                self.instr_dag.add_start(r, buffer, index, ref)

    def __enter__(self):
        global _current_program
        if _current_program != None:
            raise RuntimeError("There is already a MSCCL Program in context")
        logging.info("Entering MSCCLProgram context.")
        _current_program = self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _current_program
        if _current_program != self:
            raise RuntimeError("This program is not currently in context")
        logging.info("Exiting MSCCLProgram context.")
        _current_program = None

    def apply_send(self, src, src_buffer, src_index, dst, dst_buffer, dst_index, size):
        logging.debug(f"Applying send from rank {src} to {dst} for size {size}")
        src_buffer, src_index = self.collective.get_buffer_index(src, src_buffer, src_index)
        dst_buffer, dst_index = self.collective.get_buffer_index(dst, dst_buffer, dst_index)
        sb = self.buffers[src][src_buffer]
        db = self.buffers[dst][dst_buffer]
        for i in range(size):
            db[dst_index + i] = sb[src_index + i]

    def apply_reduce(self, src, src_buffer, src_index, dst, dst_buffer, dst_index, size):
        logging.debug(f"Applying reduce from rank {src} to {dst} for size {size}")
        src_buffer, src_index = self.collective.get_buffer_index(src, src_buffer, src_index)
        dst_buffer, dst_index = self.collective.get_buffer_index(dst, dst_buffer, dst_index)
        sb = self.buffers[src][src_buffer]
        db = self.buffers[dst][dst_buffer]
        for i in range(size):
            reduce_chunk = db[dst_index + i]
            sent_chunk = sb[src_index + i]
            db[dst_index + i] = reduce_chunk.reduce(dst, sent_chunk)

    def get_ref(self, rank, buffer, index, size):
        logging.debug(f"Getting reference for rank {rank}, buffer {buffer}, index {index}, size {size}")
        buffer, index = self.collective.get_buffer_index(rank, buffer, index)
        return Ref(rank, buffer, index, size, self)

    def lower(self):
        logging.info("Lowering program to XML")
        self.instr_dag.convert_set_list()
        if self.instr_fusion:
            logging.info("Optimizing instruction DAG")
            self.instr_dag.optimize()
        self.instr_dag._complete_metadata()
        if self.threadblock_policy == ThreadblockPolicy.manual:
            logging.info("Manually assigning threadblocks")
            manual_assign_tbs(self.instr_dag)
        else:
            logging.info("Auto-assigning threadblocks")
            auto_assign_tbs(self.instr_dag)
        self.instr_dag.lower_pt1(self.instances)
        gpu_prgms = self.instr_dag.lower_pt2(self.instances, self.interleaved_replication)
        if self.check_xml:
            logging.info("Checking XML for dependencies and threadblock order")
            check_dependency_cycles(self.instr_dag.tbs)
            check_threadblock_ordering(self.instr_dag)
        return Program(self.name, self.collective.name, self.collective.inplace, self.protocol, gpu_prgms)

    def generate_xml(self):
        logging.info(f"Generating XML for program {self.name}")
        return ir_to_xml(self.lower(), dependence_nop=self.dependence_nop)

def XML():
   logging.info("Generating and printing XML.")
   print(_curr().generate_xml())

def Check():
    logging.info("Checking the program.")
    return _curr().check()

@dataclass
class Ref(ChunkRef):
    prog: MSCCLProgram

    def copy(self, dst, buffer=None, index=-1, sendtb=-1, recvtb=-1, ch=-1):
        logging.debug(f"Copying chunk from rank {self.rank} to {dst}")
        self.prog.check_buffer_exists(dst, buffer)

        if index == -1 and buffer == None:
            index = self.index
            buffer = self.buffer
        elif index == -1 and buffer is not Buffer.input and buffer is not Buffer.output:
            index = self.prog.buffers[dst][buffer].instance_size()

        buffer, index = self.prog.collective.get_buffer_index(self.rank, buffer, index)
        dst_chunkref = self.prog.get_ref(dst, buffer, index, self.size)

        if dst_chunkref == self:
            return

        self.prog.apply_send(self.rank, self.buffer, self.index, dst, buffer, index, self.size)

        sender = self.rank
        receiver = dst
        if sender != receiver:
            logging.debug(f"Adding send operation from {sender} to {receiver}")
            sop = self.prog.instr_dag.add_send(sender, self, dst_chunkref, sendtb, ch)
            rop = self.prog.instr_dag.add_recv(receiver, self, dst_chunkref, recvtb, ch, sop)
            sop.recv_match = rop
        else:
            self.prog.instr_dag.add_copy(sender, self, dst_chunkref, sendtb, ch)

        return dst_chunkref

    def reduce(self, other_chunkref, sendtb=-1, recvtb=-1, ch=-1):
        logging.debug(f"Reducing chunk from rank {other_chunkref.rank} into rank {self.rank}")
        dst = self.rank
        src = other_chunkref.rank

        assert (self.prog.topo.link(src, dst) or src == dst), f'No link from {src} to {dst}'
        self.prog.apply_reduce(src, other_chunkref.buffer, other_chunkref.index, dst, self.buffer, self.index, self.size)

        if src != dst:
            logging.debug(f"Adding reduce operation from {src} to {dst}")
            sop = self.prog.instr_dag.add_send(src, other_chunkref, self, sendtb, ch)
            rop = self.prog.instr_dag.add_recv_reduce_copy(dst, other_chunkref, self, recvtb, ch, sop)
            sop.recv_match = rop
        else:
            self.prog.instr_dag.add_reduce(src, other_chunkref, self, sendtb, ch)

        return self
