
% startup script to make Octave/Matlab aware of the gptp_multi_output

disp(['Adding path, please run startup.m in GPML3.6 first...']);

OCT = exist('OCTAVE_VERSION') ~= 0;           % check if we run Matlab or Octave

me = mfilename;                                            % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
if OCT && numel(mydir)==2 
  if strcmp(mydir,'./'), mydir = [pwd,mydir(2:end)]; end
end                 % OCTAVE 3.0.x relative, MATLAB and newer have absolute path

addpath(mydir(1:end-1))
addpath([mydir,'cov'])
addpath([mydir,'example'])
addpath([mydir,'init'])
addpath([mydir,'sample'])
addpath([mydir,'solver'])
addpath([mydir,'util'])

clear me mydir