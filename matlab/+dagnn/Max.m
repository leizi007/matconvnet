classdef Max < dagnn.ElementWise
  %MAX DagNN max layer
  %   Max layer takes the max of all its inputs and store the result as its
  %   only output.
  
  properties (Transient)
      numInputs
      I
  end
  
  methods
      function outputs = forward(obj, inputs, params)
          obj.numInputs = numel(inputs);
          data = cat(5, inputs{:});
          [outputs{1}, obj.I] = max(data, [], 5);          
      end
      
      function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
          for k = 1 : obj.numInputs
              derInputs{k} = derOutputs{1} .* (obj.I == k);
          end
          derParams = {};
      end
      
      function reset(obj)
          obj.I = [];
      end
      
      function outputSizes = getOutputSizes(obj, inputSizes)
          outputSizes{1} = inputSizes{1};
          for k = 2 : numel(inputSizes)
              if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
                  if ~isequal(inputSizes{k}, outputSizes{1})
                      warning('Max layer: the dimensions of the input variables is not the same/');
                  end
              end
          end
      end
      
      function rfs = getReceptiveField(obj)
          numInputs = numel(obj.net.layers(obj.layerIndex).inputs);
          rfs.size = [1 1];
          rfs.stride = [1 1];
          rfs.offset = [1 1];
          rfs = repmat(rfs, numInputs, 1);
      end
      
      function obj = Sum(varargin)
          obj.load(varargin);
      end
  end
end 
      
      
      
      