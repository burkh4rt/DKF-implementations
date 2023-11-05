classdef DiscriminativeKalmanFilter < handle
    properties
        stateModelA;
        stateModelGamma;
        stateModelS;
        measurementModelF;
        measurementModelQ;
        currentPosteriorMean;
        currentPosteriorCovariance;
        dState;
    end
    
    methods
        function obj = DiscriminativeKalmanFilter(stateModelA, ...
                stateModelGamma, stateModelS, measurementModelF, ...
                measurementModelQ, currentPosteriorMean, ...
                currentPosteriorCovariance)
            obj.stateModelA = stateModelA;
            obj.stateModelGamma = stateModelGamma;
            obj.stateModelS = stateModelS;
            obj.measurementModelF = measurementModelF;
            obj.measurementModelQ = measurementModelQ;
            obj.currentPosteriorMean = currentPosteriorMean;
            obj.currentPosteriorCovariance = currentPosteriorCovariance;
            obj.dState = size(stateModelA,1);
            assert(all(size(obj.stateModelA) == [obj.dState,obj.dState]));
            assert(all(size(obj.stateModelGamma) == ...
                [obj.dState,obj.dState]));
            assert(all(eig(obj.stateModelGamma) > 0));
            assert(issymmetric(obj.stateModelGamma));
            assert(all(size(obj.stateModelS) == [obj.dState,obj.dState]));
            assert(all(eig(obj.stateModelS) > eps));
            assert(issymmetric(obj.stateModelS));
            assert(all(size(obj.currentPosteriorMean) == [obj.dState,1]));
            assert(all(size(obj.currentPosteriorCovariance) == ...
                [obj.dState,obj.dState]));
            assert(all(eig(obj.currentPosteriorCovariance) > eps));
            assert(issymmetric(obj.currentPosteriorCovariance));
        end
        
        function stateUpdate(obj)
            obj.currentPosteriorMean = ...
                obj.stateModelA * obj.currentPosteriorMean;
            obj.currentPosteriorCovariance = obj.stateModelA * ...
                obj.currentPosteriorCovariance * obj.stateModelA' ...
                + obj.stateModelGamma;
        end
        
        function measurementUpdate(obj, newMeasurement)
            Qx = feval(obj.measurementModelQ,newMeasurement);
            fx = feval(obj.measurementModelF,newMeasurement);
        	assert(all(size(fx) == [obj.dState,1]));
        	assert(all(size(Qx) == [obj.dState,obj.dState]));
        	assert(all(eig(Qx) > eps));
        	assert(issymmetric(Qx));
            if ~all(eig(inv(Qx) - inv(obj.stateModelS)) > eps)
                Qx = inv(inv(Qx) + inv(obj.stateModelS));
            end
            newPosteriorCovInv = ...
                inv(obj.currentPosteriorCovariance) + inv(Qx) ...
                - inv(obj.stateModelS);
        	obj.currentPosteriorMean = newPosteriorCovInv \ ...
                (obj.currentPosteriorCovariance\obj.currentPosteriorMean...
                + Qx\fx);
            obj.currentPosteriorCovariance = inv(newPosteriorCovInv);
        end
        
        function currentPosteriorMean = predict(obj, newMeasurement)
            stateUpdate(obj);
            measurementUpdate(obj, newMeasurement);
            currentPosteriorMean = obj.currentPosteriorMean;
        end
    end
end
