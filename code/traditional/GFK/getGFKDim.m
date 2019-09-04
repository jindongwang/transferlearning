% Copyright (c) 2013, Basura Fernando
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without modification, 
% are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice, this 
%list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice, 
% this list of conditions and the following disclaimer in the documentation and/or 
% other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
% ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% This is the subspace dimensionality estimation method proposed in
% B. Gong, Y. Shi, F. Sha, and K. Grauman. Geodesic flow kernel for unsupervised domain adaptation. In CVPR, 2012.
% 
% Please cite
% 
% @inproceedings{Fernando2013b,
% author = {Basura Fernando, Amaury Habrard, Marc Sebban, Tinne Tuytelaars},
% title = {Unsupervised Visual Domain Adaptation Using Subspace Alignment},
% booktitle = {ICCV},
% year = {2013},
% } 
%
function RES = getGFKDim(Source_Data,Target_Data,Pss,Pts,Psstt)

if nargin < 3
    [Pss,D1,E1] = princomp(Source_Data);
end
if nargin < 4
    [Pts,D2,E2] = princomp(Target_Data);
end
if nargin < 5
    [Psstt,D2,E2] = princomp([Source_Data;Target_Data]);
end

    DIM = round(size(Source_Data,2) * 0.5) ;  % if the dimensionality is large set DIM to say 200    
    RES = -1;
    for d = 1 : DIM
        Ps = Pss(:,1:d);
        Pt = Pts(:,1:d);
        Pst = Psstt(:,1:d);
        alpha1 = getAngle(Ps,Pst,d);
        alpha2 = getAngle(Pt,Pst,d);
        D = (alpha1 + alpha2)*0.5;
        for dd = 1 : d
            if(round(D(1,dd)*100) == 100)                
                RES =d;
                return;
            end
        end        
    end
end

function alpha = getAngle(Ps,Pt,DD)

    Q = [Ps, null(Ps')];
    N = size(Q,2); % N is the
    dim = size(Pt,2);
    QPt = Q' * Pt;
    [V1,V2,V,Gam,Sig] = gsvd(QPt(1:dim,:), QPt(dim+1:end,:)); 
    alpha = zeros(1,DD);
    for i = 1 : DD
        alpha(1,i) =sind(real(acosd(Gam(i,i))));
    end
end
