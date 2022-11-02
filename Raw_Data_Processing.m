clear all, close all, clc;

fn = 'C:\Users\amoadi\Desktop\RFBeam\September_12_2022\Chandler\1m_Breathing\SigUWB.txt';

data = importdata(fn);

l = size(data,1);

I_Data = data(1:1:l/2,:);
Q_Data = data(l/2+1:1:end,:);

IQ_Data = I_Data + 1j*Q_Data;

bin_length = 8 * 1.5e8/23.328e9;
max_range = 192*bin_length;

IQ_Data(1:12,:) = 0;

test = ifft(IQ_Data,[],2);
test = ifftshift(test,2);

test2 = fft(test,[],2);
test2 = fftshift(test2,2);

Y = fft(IQ_Data,[],2);
Y = fftshift(Y,2);

G = fft(IQ_Data,[],1);
G = fftshift(G,1);

T = fft(G,[],2);
T = fftshift(T,2);

%% Testing
c = 2.99792458e+08;             % speed of light, m/s 
lambda = c/7.29e9;               % radarWavEffength in m
fs = 23.328e9;                 % rangeSamplingRate in Hz 
tau = 3.7120e-005;             % rangePulseLength in sec
prf = 143e6/16;             % PRF in Hz
rangeGateDelay = 0.0055;       % Range Gate Delay in sec
ro = 0.18;

sumLines = sum(IQ_Data(2:end,:) .* conj(IQ_Data(1:end-1,:)));
avgPhChg = atan2(imag(sumLines),real(sumLines));
fdc = mean(avgPhChg)/(2*pi)*prf;

v = 0.031;
vEff = v;
del_sr = 8 * c/2/23.328e9;
nValid = size(IQ_Data,1);
numLines = size(IQ_Data,2);
validAzPts = 405;
doppler = fdc;

%delta_f = [0:numLines-1]' * (prf/validAzPts);
%offset = (1./sqrt(1-(lambda*delta_f/ (2*vEff) ).^2) - 1)* (ro + [0:nValid-1] * del_sr );
%offset = round(offset/del_sr);  % pixels offset

dCr = 1.24/size(IQ_Data,2);
crossRangeBins = [-202:1:202];
downRangeBins = [0:1:180-1];

x0 = round(size(IQ_Data,2)/2)*dCr;
y0 = floor(size(IQ_Data,1)/2)*bin_length;
V = 0.0031;
T = 1;
k = crossRangeBins;

delTau = 2*(sqrt(x0^2+(y0-abs(k)*T*V).^2)-x0)/c;





%% Plotting


figure;
plot(abs(Y))

figure;
plot(abs(IQ_Data))

figure;
plot(abs(Y(70,:)))

figure;
plot(angle(IQ_Data(70,:)))

figure;
subplot(222)
wBG = 20*log10(abs(test));
wBG = flip(rot90(wBG,2));
%y = ((-Cr/2)+dCr):dCr:Cr/2;
img = imagesc(1:405,1:180,wBG); colormap jet; c=colorbar; c.Label.String='dB';
caxis(max(caxis)+[-30,0]);
% caxis([-55, -30]);
hax = gca;
hax.YDir = 'normal';
ylabel('Down-Range [m]');  %ylim([0.2,4.8774]);
xlabel('Cross-Range [m]'); %xlim([-Cr/2+dCr,Cr/2]); 
%xticks([-0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6]);
%yticks([0 0.5 1 1.5 2 2.5 3 3.5 4 4.5]);
title('Test');

subplot(221)
wBG = 20*log10(abs(IQ_Data));
wBG = flip(rot90(wBG,2));
%y = ((-Cr/2)+dCr):dCr:Cr/2;
img = imagesc(1:405,1:180,wBG); colormap jet; c=colorbar; c.Label.String='dB';
caxis(max(caxis)+[-30,0]);
% caxis([-55, -30]);
hax = gca;
hax.YDir = 'normal';
ylabel('Down-Range [m]');  %ylim([0.2,4.8774]);
xlabel('Cross-Range [m]'); %xlim([-Cr/2+dCr,Cr/2]); 
%xticks([-0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6]);
%yticks([0 0.5 1 1.5 2 2.5 3 3.5 4 4.5]);
title('IQ_Data');

subplot(223)
wBG = 20*log10(abs(test2));
wBG = flip(rot90(wBG,2));
%y = ((-Cr/2)+dCr):dCr:Cr/2;
img = imagesc(1:405,1:180,wBG); colormap jet; c=colorbar; c.Label.String='dB';
caxis(max(caxis)+[-30,0]);
% caxis([-55, -30]);
hax = gca;
hax.YDir = 'normal';
ylabel('Down-Range [m]');  %ylim([0.2,4.8774]);
xlabel('Cross-Range [m]'); %xlim([-Cr/2+dCr,Cr/2]); 
%xticks([-0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6]);
%yticks([0 0.5 1 1.5 2 2.5 3 3.5 4 4.5]);
title('Test 2');

subplot(224)
wBG = 20*log10(abs(T));
wBG = flip(rot90(wBG,2));
%y = ((-Cr/2)+dCr):dCr:Cr/2;
img = imagesc(1:405,1:180,wBG); colormap jet; c=colorbar; c.Label.String='dB';
caxis(max(caxis)+[-30,0]);
% caxis([-55, -30]);
hax = gca;
hax.YDir = 'normal';
ylabel('Down-Range [m]');  %ylim([0.2,4.8774]);
xlabel('Cross-Range [m]'); %xlim([-Cr/2+dCr,Cr/2]); 
%xticks([-0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6]);
%yticks([0 0.5 1 1.5 2 2.5 3 3.5 4 4.5]);
title('FFT');

img = figure;
wBG = 20*log10(abs(test));
wBG = flip(rot90(wBG,2));
%y = ((-Cr/2)+dCr):dCr:Cr/2;
img = imagesc(1:405,1:180,wBG); colormap jet; c=colorbar; c.Label.String='dB';
caxis(max(caxis)+[-30,0]);
% caxis([-55, -30]);
hax = gca;
hax.YDir = 'normal';
ylabel('Down-Range');  %ylim([0.2,4.8774]);
xlabel('Cross-Range'); %xlim([-Cr/2+dCr,Cr/2]); 
%xticks([-0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6]);
%yticks([0 0.5 1 1.5 2 2.5 3 3.5 4 4.5]);
title('Range Compressed');
saveas(img,'RangeCompressed.png');

%{
figure;
plot(abs(IQ_Data)/max(abs(IQ_Data)));
hold on;
plot(abs(Captured_Data)/max(abs(Captured_Data)))
%}

%{
Y1 = fft(data);
Y1(60:1410,:) = 0;
Y3 = ifft(Y1);
L1 = length(Y1);

P2 = abs(Y1/L1);
P1 = P2(1:L1/2+1);
P1(2:end-1) = 2*P1(2:end-1);

fs = 23.328e9;
Fs = fs/8;

f = fs*(0:(L1/2))/L1;

figure;
plot(abs(Y3))

figure;
plot(f,P1)
%}

