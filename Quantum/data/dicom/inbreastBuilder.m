%             INbreast Release 1.0
%Breast Research Group, INESC Porto, Portugal
%http://medicalresearch.inescporto.pt/breastresearch/
%         medicalresearch@inescporto.pt 
xlsdata= read_mixed_csv('../INBreast.csv',';');
biradsclass = xlsdata(2:end,8);
biradsclassFilename=xlsdata(2:end,6);
date=xlsdata(2:end,5);
[sortedFilenames,idx]=sort(biradsclassFilename);
biradsclass=biradsclass(idx);
date=date(idx);
DIR = dir('.\*.dcm');
nn=numel(DIR);
INbreast=cell(nn,8);
for k=1:nn
    disp(num2str(k))
    info=dicominfo(DIR(k).name);
    INbreast{k,1}=k;
    
    lineData = textscan(DIR(k).name, '%s', 'Delimiter','_');
    lineData = lineData{1};
    %INbreast{k,2} is empty - name
    INbreast{k,3}=lineData{2};%patient ID
    INbreast{k,4}=lineData{3}; %Modality - MG
    INbreast{k,5}=lineData{5}; %ViewPosition CC or ML(O)
    %INbreast{k,6} is empty - date
    INbreast{k,7}=DIR(k).name;
    %INbreast{k,8} is empty - birads
end
dicomFilename=INbreast(:,7);
[sortedFilenames,idx]=sort(dicomFilename);
INbreast=INbreast(idx,:);
INbreast(:,8)=biradsclass;
INbreast(:,6)=date;

%organize by cases: a case is the set of all views for a single patient for
%a single day. We have 117 cases. 
%IMPORTANT NOTE: in the Academic Radiology publication the number is wrongly presented as 115.
PatientInfo=strcat(INbreast(:,3),INbreast(:,6));
uniquePatientInfo=unique(PatientInfo);
for n=1:length(uniquePatientInfo)
    idx = find(ismember(PatientInfo, uniquePatientInfo(n))==1);
    INbreastCase{n}=idx;
end

%just usage example
caseBirads =-10*ones(length(INbreastCase),1);
caseLen = zeros(length(INbreastCase),1);
for n=1:length(INbreastCase)
    disp(['case ', num2str(n)]); 
    curr_case = INbreastCase{n};
    curr_biradsSet=[];
    caseLen(n)=length(curr_case);
    for i=1:length(curr_case)
        filename = INbreast{curr_case(i),7};
        birads= INbreast{curr_case(i),8};
        curr_biradsSet=[curr_biradsSet,sscanf(birads, '%g')];
        
        %img=dicomread(filename);
        %imshow(img,[]);
    end
    caseBirads(n)=max(curr_biradsSet);
end
%[caseLen caseBirads]
figure, hist(caseLen, 1:8)
figure, hist(caseBirads, 0:6)



