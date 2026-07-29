%%% This code was developed in Anantram's group at the University of Washington
%%% by Hashem Mohammad, Jianqing Qi et. al. Please send questions to anantmp@uw.edu and hashemm@uw.edu.

%11-19-2018%
%Decoherent Transmission for DNA strands


function  DNATransmission_Decoherence(run_num)
%% Initialization (Loads Parameters from subdir run_num)
format long

apath = strcat(pwd,'/');
addpath(apath)

d=['run' num2str(run_num)];
dir_path=  strcat(apath,d,'/');    %loads subdirectory
addpath(dir_path)
workdir = dir_path;
disp(workdir);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Specify the file name
file = 'Parameters.txt';

% Read the content of the file
fid = fopen(file, 'r');
content = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
fclose(fid);

% Remove empty lines
content = content{1};  % Extract the cell array from the textscan output
nonEmptyLines = ~cellfun('isempty', content);
content = content(nonEmptyLines);

% Write the modified content back to the file
filepath = fullfile(workdir,file);
fid = fopen(filepath, 'w');
fprintf(fid, '%s\n', content{:});
fclose(fid);

clearvars content nonEmptyLines fid file filepath
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x=fopen('Parameters.txt');
dataArray=textscan(x,'%s','WhiteSpace','\r\n');
loc1=find(~cellfun(@isempty,strfind(dataArray{1,1},'Energy')));
strand=char(dataArray{1,1}(1));
Orbitals=dataArray{1,1}(3:loc1-1);
Orbitals=cellfun(@str2num,Orbitals)';

loc2=find(~cellfun(@isempty,strfind(dataArray{1,1},'Inject')));
Energy=dataArray{1,1}(loc1+1:loc2-1);
Energy=cellfun(@str2num,Energy)';
loc1=loc2;

loc2=find(~cellfun(@isempty,strfind(dataArray{1,1},'Extract')));
Lsite=dataArray{1,1}(loc1+1:loc2-1);
Lsite=cellfun(@str2num,Lsite)';
loc1=loc2;

loc2=find(~cellfun(@isempty,strfind(dataArray{1,1},'GammaL')));
Rsite=dataArray{1,1}(loc1+1:loc2-1);
Rsite=cellfun(@str2num,Rsite)';

loc1=loc2;
loc2=find(~cellfun(@isempty,strfind(dataArray{1,1},'GammaR')));
gammaL=dataArray{1,1}(loc1+1:loc2-1);
gammaL=str2double(cell2mat(gammaL))';
loc1=loc2;

loc2=find(~cellfun(@isempty,strfind(dataArray{1,1},'Probes')));
gammaR=dataArray{1,1}(loc1+1:loc2-1);
gammaR=str2double(cell2mat(gammaR));
loc1=loc2;

loc2=find(~cellfun(@isempty,strfind(dataArray{1,1},'Broadening')));
Dsites=dataArray{1,1}(loc1+1:loc2-1);
Dsites=cellfun(@str2num,Dsites)';

eta=dataArray{1,1}(loc2+1);
eta=str2double(cell2mat(eta));
bprobe=dataArray{1,1}(loc2+3:end);
bprobe=cellfun(@str2num,bprobe)';

fclose(x);
clearvars loc1 loc2 dataArray x

%%%%%%%%%%%%%Load Matrices%%%%%%%%%%%%%%%%%%%%%%%%%%
%Fm = ['load ',apath, strand, '.mat'];
Fm = ['load ', strand,'.mat'];
disp(Fm);
eval(Fm);
H0 = eval(strand);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eta = 0;
sizeH = size(H0, 1);  % size of Hamiltonian
disp(['Size of the Hamiltonian = ' num2str(sizeH)])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Norb = sum(Orbitals);    %total size of H,total number of orbitals
if(Norb ~= sizeH)
    disp('Total size of H =/= number of Orbitals, please check your input file')
    return;
end
Nb = length(Dsites);
Nsite = length(Orbitals);   %total number of atoms

%%%%%%%%%%%Initialize gamma %%%%%%%%%%%%%%
sites = [Lsite Rsite Dsites];
gamma = [gammaL*ones(1,length(Lsite)) gammaR*ones(1,length(Rsite)) bprobe];

%%%%%%%%%set sumSig %%%%%%%%%%%%%%%%%%%%%
sumSig = zeros(sizeH, sizeH);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ii = 1 : Nsite
    isite = sites(ii);
    TempLen1 = sum(Orbitals(1 : isite)) - Orbitals(isite);
    TempLen2 = sum(Orbitals(1 : isite));
    Len = TempLen2 - TempLen1;
    sumSig(TempLen1 + 1 : TempLen2, TempLen1 + 1 : TempLen2) = gamma(ii) * eye(Len);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sumSig = -1i * sumSig / 2;
%% Loop Initialization %%
NE = length(Energy);
%%%%%%%% Check available files from Checkpoint %%%%%%%%%%%
mat=dir([workdir 'Teff_*.mat']);

if length(mat)~=0
load(mat.name);
qq=find(T~=-1, 1, 'last' )+1;
else
    T=-1*ones(1,NE);
    qq=1;
end

%%%prepare the output name before entering the loop%%%
Tname=strcat(workdir,'Teff_',strand,'_gammaL_',num2str(gammaL),'_gammaR_',num2str(gammaR),'_bprobe_',num2str(mean(bprobe)),'.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Entering the loop 

for nE = qq : NE
    E = Energy(nE);
    Gr = ((E + 1i * eta) * eye(sizeH) - H0 - sumSig) \ eye(sizeH);
    Ga = Gr';

%%%%%%%%%% Transmission %%%%%%%%%%%%%
%%%%%%%%%% transmission between every 2 probes %%%%%%%%%%%%%
Tmat = zeros(Nsite, Nsite);

for ii =  2: Nsite
    isite = sites(ii);
    TempLeni1 = sum(Orbitals(1 : isite)) - sum(Orbitals(isite));
    TempLeni2 = sum(Orbitals(1 : isite));
    Leni = TempLeni2 - TempLeni1;
    Gammai = gamma(ii) * eye(Leni);

    for jj = 1 : (ii - 1)
        jsite = sites(jj);
        TempLenj1 = sum(Orbitals(1 : jsite)) - sum(Orbitals(jsite));
        TempLenj2 = sum(Orbitals(1 : jsite));
        Lenj = TempLenj2 - TempLenj1;
        Gammaj = gamma(jj) * eye(Lenj);

        Tmat(ii, jj) = real(trace(Gammai * Gr(TempLeni1 + 1 : TempLeni2, TempLenj1 + 1 : TempLenj2) * Gammaj * Ga(TempLenj1 + 1 : TempLenj2, TempLeni1 + 1 : TempLeni2)));
        Tmat(jj, ii) = Tmat(ii, jj);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LL = length([Lsite Rsite]);

TL = [];
TR = [];
for ii = LL+1 : Nsite         %include only Bprobe sites (no GammaL/GammaR)
    TL = [TL Tmat(1:length(Lsite), ii)];    %Lsite-to-Bprobe coupling
    TR = [TR; Tmat(ii, length(Lsite)+1:length(Lsite)+length(Rsite))]; %Rsite-to-Bprobe coupling
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Wmat = zeros(Nb);

for ii = 1 : Nb
    
        Wmat(ii, ii) = sum(Tmat(ii + LL, :));
        
    for jj = ii + 1 : Nb
        Wmat(ii, jj) = -Tmat(ii + LL, jj + LL);
    end
end

Wmat = Wmat' + Wmat;
for ii = 1 : Nb 
    Wmat(ii, ii) = Wmat(ii, ii) / 2;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Td = Tmat(1:length(Lsite), length(Lsite)+1:length(Lsite)+length(Rsite)) + TL * (Wmat \ eye(Nb)) * TR;
T(nE) = sum(sum(Td));
%%%%%%% Save after every iteration %%%%
save(Tname,'Energy','T')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


disp('Finished Decoherent Transmission!')
