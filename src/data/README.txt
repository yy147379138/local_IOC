Explaination for data:

This folder contains all the trajectory files as inputs to training or forecasting .exe program

./data/hires_training_corrected.trajectory 
Original data collected in Intel lab, given by Brain Ziebart. Usually this
is input to train.exe.

./data/trainSet.trajectory 
A virtual robot position is added to the end of each trajectory. The evidence
class reads the last point of the path as the static robot.This is used to train
non-robot ended paramters in discrete MDP based model by trainplus.exe. 
In addition, this data is used to train non-robot ended cost matrix in LQ model
by trainLQ.exe

./data/robset.trajectory 
A virtual robot position is added to the end of each trajectory. Median point is
inserted between two consective original points along the path to fake low velocity
distribution. This is used to train robot ended paramters by trainplus.exe.

./data/testSet.trajectory
The set contains 15 trajectories, some are from robset and other are from trainSet.
Tested by forecastplus.exe

./data/testSet2.trajectory
The set contains 5 trajectories, some are from robset and other are from trainSet.
Tested by forecastplus.exe

./data/testlqIntel.trajectory 
The set contains several trajectories from Intel data to test linear-quadratic model.
Tested by lqforecast.exe

./data/lqToy.trajectory
A approximated quadratic curve trajectory to test the lq model.

./data/writeTraj.cpp
This code is writes a single toy trajectory file

All .sicktraj files: trajectory collected@NSH1507
***********************************************************************************************
rob.sicktraj: robot-ended trajectory

nonrob.sicktraj: non-rob-ended trajectory

test1.sicktraj: specially concatenated testing trajectories

test2.sicktraj: person6 trajectories




