import tensorflow as tf
import numpy as np
import struct
import os

PAGESIZE=4096

class OffsetTableBuilder:
    def __init__(self, tfr_filename, padding=False):
        self.padding=padding
        self.tfr_filename=tfr_filename
        self.offset_table=[]

    # this way is more convenient than actually calculating the offset
    def calc_offset(self, serialized_instance, writer):
        if(self.padding):
            current_filesize=os.path.getsize(self.tfr_filename)
            cur_page=int(current_filesize/PAGESIZE)
            calc_new_filesize=current_filesize+len(serialized_instance)+16+16
            calc_new_page=int(calc_new_filesize/PAGESIZE)            
            
            if(calc_new_page>cur_page):
                # if exceeds: padding until reaching new page
                #TODO: the pad still needs 16B metadata
                pad_len=calc_new_page*PAGESIZE-current_filesize
                pad=b''
                if(pad_len-16>0):                    
                    for _ in range (pad_len-16):
                        pad+=b'0'
                    # TODO: 不該在oftbuilder動tfrfile 好醜
                    self.tfr_filename
                    writer.write(pad)
                else:
                    print("error")
        current_filesize=os.path.getsize(self.tfr_filename) 
        self.offset_table.append(current_filesize)
        
    def write(self, oft_filename):
        oft=open(oft_filename,'wb')
        for offset in self.offset_table:
            oft.write(offset.to_bytes(8, byteorder = 'little'))
        oft.close()