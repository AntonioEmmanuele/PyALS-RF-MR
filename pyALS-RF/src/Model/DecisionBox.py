"""
Copyright 2021-2023 Salvatore Barone <salvatore.barone@unina.it>

This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
RMEncoder; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""
import copy
from enum import Enum
from pyalslib import double_to_hex, apply_mask_to_double, apply_mask_to_int
import numpy as np
import struct

class DecisionBox:
  class CompOperator(Enum):
    lessThan = 1
    equal = 2
    greaterThan = 3
    
  def __init__(self, box_name = None, feature_name = None, data_type = None, operator = None, threashold = None, nab = 0):
    self.name = box_name
    self.feature_name = feature_name
    self.data_type = data_type
    if operator:
      if operator == "greaterThan":
          self.operator = DecisionBox.CompOperator.greaterThan
      elif operator == "lessThan":
        self.operator = DecisionBox.CompOperator.lessThan
      elif operator == "equal":
        self.operator = DecisionBox.CompOperator.equal
      else:
        raise Exception("Sorry, operator not recognized") 
    self.threshold_string = threashold
    self.threshold = np.float32(threashold)
    self.nab = int(nab)
    # self.compare_many = self.inner_compare_many
    # self.fixed_value = None
    self.thd_back = self.threshold

  def __deepcopy__(self, memo = None):
    box = DecisionBox()
    box.name = copy.deepcopy(self.name)
    box.feature_name = copy.deepcopy(self.feature_name)
    box.data_type = copy.deepcopy(self.data_type)
    box.operator = copy.deepcopy(self.operator)
    box.threshold = copy.deepcopy(self.threshold)
    box.thd_back = copy.deepcopy(self.thd_back)
    box.threshold_string = copy.deepcopy(self.threshold_string)
    box.nab = copy.deepcopy(self.nab)
    return box

  def get_c_operator(self):
    if self.operator == DecisionBox.CompOperator.greaterThan:
      return ">"
    elif self.operator == DecisionBox.CompOperator.lessThan:
      return "<"
    else: 
      return "==" 
  
  def get_str_op(self):
    if self.operator == DecisionBox.CompOperator.greaterThan:
      return "greaterThan"
    elif self.operator == DecisionBox.CompOperator.lessThan:
      return "lessThan"
    else: 
      return "equal" 
    
  def get_hexstr_threashold(self):
    if self.data_type == "f32":
      return str(double_to_hex(self.threshold))[2:]
    else:
      return hex(int(self.threshold))[2:]

  def get_struct(self):
    c_operator = "=="
    operator = "equals"
    if self.operator == DecisionBox.CompOperator.greaterThan:
      c_operator = ">"
      operator = "greaterThan"
    elif self.operator == DecisionBox.CompOperator.lessThan:
      c_operator = "<"
      operator = "lessThan"
    threshold = str(self.threshold)
    hex_threshold = ""
    if self.data_type == "f32":
      #hex_threshold = str(double_to_hex(float(self.threshold)))[2:]
      hex_threshold = struct.pack('!f', float(self.threshold)).hex()
      data_type = "f32"
    elif self.data_type == "int16":
      threshold = int(float(self.threshold))         # string → int
      val = np.int16(threshold)               # restrict to 16 bit
      hex_threshold = format(np.uint16(val), '04x')   # 4 digit hex
      data_type = "int"
    else: 
      hex_threshold = hex(int(self.threshold))[2:]
    return {"name"          : self.name,
            "feature"       : self.feature_name,
            "data_type"     : data_type,
            "operator"      : operator,
            "c_operator"    : c_operator,
            "threshold"     : threshold,
            "threshold_hex" : hex_threshold}


  def get_data_width(self):
    if self.data_type == "f32":
      return 32
    else:
      return 16
    
  def compare(self, input):
    if self.data_type == "f32":
    
      # # Both the input value and the threshold are masked according to the configured number of approximate bits before
      # # the comparison takes place.
      # if self.nab == 0:
      #   # Whether no approximation is required, input and threshold are simply converted to the suitable data-type.
      #   input_to_compare = float(input) 
      #   threshold = float(self.threshold)
      # else:
      #   input_to_compare = apply_mask_to_double(float(input), self.nab) 
      #   threshold = apply_mask_to_double(float(self.threshold), self.nab)
      threshold = np.float32(float(self.threshold))
      input_to_compare = np.float32(float(input))

    # elif self.nab != 0:
    #   input_to_compare = apply_mask_to_int(int(input), self.nab) 
    #   threshold = apply_mask_to_int(int(self.threshold), self.nab)
    
    # This is for adding support to int16 quantization.
    elif self.data_type ==  "int16":
        input_to_compare  = np.int16(np.round(input))
        threshold         = np.int16(np.round(float(self.threshold)))
    elif self.data_type == "int8":
        input_to_compare  = np.int8(np.round(input))
        threshold         = np.int8(np.round(float(self.threshold)))
    else:
      # Whether no approximation is required, input and threshold are simply converted to the suitable data-type.
      input_to_compare = int(input) 

      threshold = int(self.threshold)
    if self.operator == DecisionBox.CompOperator.greaterThan:
      return input_to_compare > threshold
    elif self.operator == DecisionBox.CompOperator.lessThan:
      return input_to_compare < threshold
    else: 
      return input_to_compare == threshold
  

  # def inner_compare_many(self, inputs):
  def compare_many(self, inputs):
    # arr = np.asarray(inputs)

    
    # if self.data_type == "f32":
    #     input_to_compare = np.float32(arr.astype(float))
    #     # threshold = np.float32(float(self.threshold))

    # elif self.data_type == "int16":
    #     input_to_compare = np.int16(np.round(arr))
    #     # threshold = np.int16(np.round(float(self.threshold)))

    # elif self.data_type == "int8":
    #     input_to_compare = np.int8(np.round(arr))
    #     # threshold = np.int8(np.round(float(self.threshold)))

    # else:
    #     # int32 / default case
    #     input_to_compare = arr.astype(int)
    #     # threshold = int(self.threshold)

    # --- Operator ----------------------------------------------------------
    if self.operator == DecisionBox.CompOperator.greaterThan:
        return inputs > self.threshold

    elif self.operator == DecisionBox.CompOperator.lessThan:
        return inputs < self.threshold

    else:
        return inputs == self.threshold

  def set_dt(self, data_type = "f32"):
    self.data_type = data_type
    if data_type == "f32":
      self.threshold = np.float32(float(self.threshold_string))
      self.thd_back = self.threshold
    elif data_type == "int16":
      self.threshold = np.int16(np.round(float(self.threshold_string)))
      self.thd_back = self.threshold 
    elif data_type == "int8":
      self.threshold = np.int8(np.round(float(self.threshold_string)))
      self.thd_back = self.threshold
    else:
      assert 1 == 0, "Invalid Feature Repr."
    
  def flip_thd_bit(self, bit_to_flip, value):
      """
      Generic stuck-at bit fault on self.threshold.
      bit_to_flip: 0 = LSB
      value: 0 or 1
      """
      # self.thd_back = self.threshold
      dtype_map = {
          "f32": np.float32,
          "int16":  np.int16,
          "int8":   np.int8,
      }
      dtype = dtype_map[self.data_type]
      thd_arr = np.array([self.threshold], dtype=dtype)

      # Unsigned view of same width
      unsigned = np.dtype(f"u{thd_arr.itemsize}")
      u = thd_arr.view(unsigned)

      mask = unsigned.type(1) << bit_to_flip

      if value:
          u[0] |= mask
      else:
          u[0] &= ~mask

      
      self.threshold = thd_arr.view(dtype)[0]


class FaultedBox:
      
  def __init__(self, box_name, feature_name, data_type, fixed_value):
    self.name = box_name
    self.feature_name = feature_name
    self.data_type = data_type
    self.fixed_value = fixed_value

  def compare(self, input):
    return self.fixed_value 
  
  def compare_many(self, inputs):
    return np.full(len(inputs), self.fixed_value, dtype=bool)

# class FaultedBoxBitLevel(DecisionBox):
  
#   def __init__(self, box_name = None, feature_name = None, data_type = None, operator = None, threashold = None, nab = 0):
#     self.name = box_name
#     self.feature_name = feature_name
#     self.data_type = data_type
#     if operator:
#       if operator == "greaterThan":
#           self.operator = DecisionBox.CompOperator.greaterThan
#       elif operator == "lessThan":
#         self.operator = DecisionBox.CompOperator.lessThan
#       elif operator == "equal":
#         self.operator = DecisionBox.CompOperator.equal
#       else:
#         raise Exception("Sorry, operator not recognized") 
#     self.threshold = threashold
#     self.nab = int(nab)
  
#   def compare_many(self, inputs):
#     """
#     Some vectorization never happens.
#     """
#     # Convertiamo inputs in array NumPy per vettorializzare tutto
#     arr = np.asarray(inputs)

#     # --- Gestione data type -------------------------------------------------
#     if self.data_type == "f32":
#         # Conversione unica per tutto il vettore
#         input_to_compare = np.float32(arr.astype(float))
#         threshold = np.float32(float(self.threshold))

#     elif self.data_type == "int16":
#         input_to_compare = np.int16(np.round(arr))
#         threshold = np.int16(np.round(float(self.threshold)))

#     elif self.data_type == "int8":
#         input_to_compare = np.int8(np.round(arr))
#         threshold = np.int8(np.round(float(self.threshold)))

#     else:
#         # int32 / default case
#         input_to_compare = arr.astype(int)
#         threshold = int(self.threshold)

#     # --- Operator ----------------------------------------------------------
#     if self.operator == DecisionBox.CompOperator.greaterThan:
#         return input_to_compare > threshold

#     elif self.operator == DecisionBox.CompOperator.lessThan:
#         return input_to_compare < threshold

#     else:
#         return input_to_compare == threshold



class FaultedBoxRegInput:
  dtype_map = {
          "f32": np.float32,
          "int16":  np.int16,
          "int8":   np.int8,
      }
  
  class CompOperator(Enum):
    lessThan = 1
    equal = 2
    greaterThan = 3
  
  def __init__(self, decision_box:DecisionBox, bit_to_flip: int, fixed_value: int ):
    self.name = decision_box.name
    self.feature_name = decision_box.feature_name
    self.data_type = decision_box.data_type 
    self.operator = decision_box.operator
    self.threshold_string = decision_box.threshold_string
    self.threshold = decision_box.threshold
    self.nab = decision_box.nab
    self.thd_back = decision_box.thd_back

    # Computing the mask for masking values 
    dtype = FaultedBoxRegInput.dtype_map[self.data_type]
    thd_arr = np.array([self.threshold], dtype=dtype)
    unsigned = np.dtype(f"u{thd_arr.itemsize}")
    u = thd_arr.view(unsigned)
    # self.mask = unsigned.type(1) << bit_to_flip
    self.mask = unsigned.type(1 << bit_to_flip)
    self.fixed_value = fixed_value
  
  def __deepcopy__(self, memo = None):
    box = FaultedBoxRegInput()
    box.name = copy.deepcopy(self.name)
    box.feature_name = copy.deepcopy(self.feature_name)
    box.data_type = copy.deepcopy(self.data_type)
    box.operator = copy.deepcopy(self.operator)
    box.threshold = copy.deepcopy(self.threshold)
    box.thd_back = copy.deepcopy(self.thd_back)
    box.threshold_string = copy.deepcopy(self.threshold_string)
    box.nab = copy.deepcopy(self.nab)
    box.mask = copy.deepcopy(self.mask)
    box.fixed_value = copy.deepcopy(self.mask)

    return box

    

  def compare_many(self, inputs):
      """
      Apply the same stuck-at fault to a COPY of inputs, then compare.
      inputs is not modified.
      """
      # print(np.dtype(inputs[0]))
      # print(np.dtype(self.mask))
      
      # --- 1) Input copy ------------------------------------
      inp = inputs.copy()

      # --- 2) Bit-level fault injection -------------------------------------
      unsigned = np.dtype(f"u{inp.itemsize}")
      u = inp.view(unsigned)

      if self.fixed_value:
          u |= self.mask
      else:
          u &= ~self.mask

      # --- 3) Compare --------------------------------------------------------
      if self.operator == DecisionBox.CompOperator.greaterThan:
          return inp > self.threshold
      elif self.operator == DecisionBox.CompOperator.lessThan:
          return inp < self.threshold
      else:
          return inp == self.threshold
    
  def set_dt(self, data_type = "f32"):
    self.data_type = data_type
    if data_type == "f32":
      self.threshold = np.float32(float(self.threshold_string))
      self.thd_back = self.threshold
    elif data_type == "int16":
      self.threshold = np.int16(np.round(float(self.threshold_string)))
      self.thd_back = self.threshold 
    elif data_type == "int8":
      self.threshold = np.int8(np.round(float(self.threshold_string)))
      self.thd_back = self.threshold
    else:
      assert 1 == 0, "Invalid Feature Repr."