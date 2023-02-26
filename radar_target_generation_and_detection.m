%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sensor Fusion Nanodegree 
% Radar Target Generation and Detection project
% Author: Kirolos Wahba Moheeb Wahba
% Date: February 26, 2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
clc;

%% Radar Specifications 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency of operation = 77GHz
% Max Range = 200m
% Range Resolution = 1 m
% Max Velocity = 100 m/s
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%speed of light = 3e8
%% User Defined Range and Velocity of target
% *%TODO* :
% define the target's initial position and velocity. Note : Velocity
% remains contant
target_position = 100;
target_velocity = 50;


%% FMCW Waveform Generation

% *%TODO* :
%Design the FMCW waveform by giving the specs of each of its parameters.
% Calculate the Bandwidth (B), Chirp Time (Tchirp) and Slope (slope) of the FMCW
% chirp using the requirements above.


%Operating carrier frequency of Radar 
fc= 77e9;             %carrier freq
maxRange = 200;
rangeResolution = 1;
lightSpeed = 3e8;
Bandwidth = lightSpeed / (2 * rangeResolution);
Tchirp = (5.5 * 2 * maxRange) / lightSpeed;
slope = Bandwidth / Tchirp;

%The number of chirps in one sequence. Its ideal to have 2^ value for the ease of running the FFT
%for Doppler Estimation. 
Nd=128;                   % #of doppler cells OR #of sent periods % number of chirps

%The number of samples on each chirp. 
Nr=1024;                  %for length of time OR # of range cells

% Timestamp for running the displacement scenario for every sample on each
% chirp
t=linspace(0,Nd*Tchirp,Nr*Nd); %total time for samples


%Creating the vectors for Tx, Rx and Mix based on the total samples input.
Tx=zeros(1,length(t)); %transmitted signal
Rx=zeros(1,length(t)); %received signal
Mix = zeros(1,length(t)); %beat signal

%Similar vectors for range_covered and time delay.
r_t=zeros(1,length(t));
td=zeros(1,length(t));


%% Signal generation and Moving Target simulation
% Running the radar scenario over the time. 

for i=1:length(t)         
    
    
    % *%TODO* :
    %For each time stamp update the Range of the Target for constant velocity. 
    r_t(i) = target_position + t(i)*target_velocity;
    td(i) = 2*r_t(i) / lightSpeed;
    
    % *%TODO* :
    %For each time sample we need update the transmitted and
    %received signal. 
    Tx(i) = cos( 2 * pi * ( fc * t(i) + slope * t(i)^2 /2 ) );
    Rx(i)  = cos( 2 * pi * ( fc * (t(i) -td(i)) + ( slope * ( t(i) - td(i) )^2) / 2 ) );
    
    % *%TODO* :
    %Now by mixing the Transmit and Receive generate the beat signal
    %This is done by element wise matrix multiplication of Transmit and
    %Receiver Signal
    Mix(i) = Tx(i) * Rx(i);
end

%% RANGE MEASUREMENT


 % *%TODO* :
%reshape the vector into Nr*Nd array. Nr and Nd here would also define the size of
%Range and Doppler FFT respectively.
Mix = reshape(Mix, [Nr, Nd]);

 % *%TODO* :
%run the FFT on the beat signal along the range bins dimension (Nr) and
%normalize.
Mix_fft = fft(Mix, [], 1);
Mix_fft = (Mix_fft - min(Mix_fft, [], 1)) ./ (max(Mix_fft, [], 1) - min(Mix_fft, [], 1));

 % *%TODO* :
% Take the absolute value of FFT output
Mix_fft = abs(Mix_fft);

 % *%TODO* :
% Output of FFT is double sided signal, but we are interested in only one side of the spectrum.
% Hence we throw out half of the samples.
Mix_fft = Mix_fft(1:Nr/2,:);

%plotting the range
figure ('Name','Range from First FFT')
subplot(2,1,1)

% *%TODO* :
% plot FFT output
plot(Mix_fft(:,1))
title('Range from First FFT (Chirp #1)')
xlabel('Range (m)')
ylabel('Amplitude')
axis ([0 200 0 1]);

subplot(2,1,2)

% plot FFT output
plot(Mix_fft(:,128))
title('Range from First FFT (Chirp #128)')
xlabel('Range (m)')
ylabel('Amplitude')
axis ([0 200 0 1]);



%% RANGE DOPPLER RESPONSE
% The 2D FFT implementation is already provided here. This will run a 2DFFT
% on the mixed signal (beat signal) output and generate a range doppler
% map.You will implement CFAR on the generated RDM


% Range Doppler Map Generation.

% The output of the 2D FFT is an image that has reponse in the range and
% doppler FFT bins. So, it is important to convert the axis from bin sizes
% to range and doppler based on their Max values.

Mix=reshape(Mix,[Nr,Nd]);

% 2D FFT using the FFT size for both dimensions.
sig_fft2 = fft2(Mix,Nr,Nd);

% Taking just one side of signal from Range dimension.
sig_fft2 = sig_fft2(1:Nr/2,1:Nd);
sig_fft2 = fftshift(sig_fft2);
RDM = abs(sig_fft2);
RDM = 10*log10(RDM) ;

%use the surf function to plot the output of 2DFFT and to show axis in both
%dimensions
doppler_axis = linspace(-100,100,Nd);
range_axis = linspace(-200,200,Nr/2)*((Nr/2)/400);
figure('Name', 'RDM From 2D FFT')
surf(doppler_axis,range_axis,RDM);
title( 'RDM From 2D FFT');
xlabel('Velocity');
ylabel('Range');
zlabel('Amplitude (dB)');
colorbar;

%% CFAR implementation

%Slide Window through the complete Range Doppler Map

% *%TODO* :
%Select the number of Training Cells in both the dimensions.
rangeTrainingCells = 10;
dopplerTrainingCells = 8;

% *%TODO* :
%Select the number of Guard Cells in both dimensions around the Cell under 
%test (CUT) for accurate estimation
rangeGuardCells = 2;
dopplerGuardCells = 4;

guardAndCUT_Cells = (2*rangeGuardCells+1)*(2*dopplerGuardCells+1);
trainingCells = (2*rangeTrainingCells+2*rangeGuardCells+1)*(2*dopplerTrainingCells+2*dopplerGuardCells+1) - guardAndCUT_Cells;

% *%TODO* :
% offset the threshold by SNR value in dB
offset = 10;

% *%TODO* :
%Create a vector to store noise_level for each iteration on training cells
noise_level = zeros(Nr/2,Nd);

%A vector to store signal after applying CA_CFAR
signal_cfar = zeros(Nr/2,Nd);

% *%TODO* :
%design a loop such that it slides the CUT across range doppler map by
%giving margins at the edges for Training and Guard Cells.
%For every iteration sum the signal level within all the training
%cells. To sum convert the value from logarithmic to linear using db2pow
%function. Average the summed values for all of the training
%cells used. After averaging convert it back to logarithimic using pow2db.
%Further add the offset to it to determine the threshold. Next, compare the
%signal under CUT with this threshold. If the CUT level > threshold assign
%it a value of 1, else equate it to 0.
for r = 1:(Nr/2-(2*rangeGuardCells+2*rangeTrainingCells))
   for d = 1:(Nd-(2*dopplerGuardCells+2*dopplerTrainingCells))
   % Use RDM[x,y] as the matrix from the output of 2D FFT for implementing
   % CFAR
   
   % CUT index
   CUT_r = r+rangeGuardCells+rangeTrainingCells+1;
   CUT_d = d+dopplerGuardCells+dopplerTrainingCells+1;
   
   noise = sum(sum(db2pow(RDM(r:r+2*rangeGuardCells+2*rangeTrainingCells,d:d+2*dopplerGuardCells+2*dopplerTrainingCells))));
   noise = noise - sum(sum(db2pow(RDM(r+rangeTrainingCells:r+rangeTrainingCells+2*rangeGuardCells,d+dopplerTrainingCells:d+dopplerTrainingCells+2*dopplerGuardCells))));
   noise_level(CUT_r, CUT_d) = pow2db(noise/trainingCells) + offset;
   
   %Compare the original signal with the noise threshold
   if RDM(CUT_r, CUT_d) > noise_level(CUT_r, CUT_d)
       signal_cfar(CUT_r, CUT_d) = 1;
   end
   end
end

% *%TODO* :
% The process above will generate a thresholded block, which is smaller 
%than the Range Doppler Map as the CUT cannot be located at the edges of
%matrix. Hence,few cells will not be thresholded. To keep the map size same
% set those values to 0. 

% Already accounted for this error in my algorithm above!


% *%TODO* :
%display the CFAR output using the Surf function like we did for Range
%Doppler Response output.
figure('Name', 'CA-CFAR Filtered RDM')
surf(doppler_axis,range_axis,signal_cfar);
% surf(doppler_axis,range_axis,noise_level);
title( 'CA-CFAR Filtered RDM');
xlabel('Velocity');
ylabel('Range');
zlabel('Normalized Amplitude');
colorbar;

%display the noise threshold using CA-CFAR for for Range
%Doppler Response output.
figure('Name', 'Noise threshold using CA-CFAR for RDM')
surf(doppler_axis,range_axis,noise_level);
title( 'Noise threshold');
xlabel('Velocity');
ylabel('Range');
zlabel('Normalized Amplitude');
colorbar;
