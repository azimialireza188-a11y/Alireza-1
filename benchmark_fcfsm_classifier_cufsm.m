function benchmark_fcfsm_classifier_cufsm(output_path)
%BENCHMARK_FCFSM_CLASSIFIER_CUFSM Export a native CUFSM 5.70 Stage-A reference.
%
% This benchmark intentionally uses a simple one-piece open channel and the
% repository's native CUFSM 5.70 functions. It validates the Python Stage-A
% implementation of K0 and fcFSM L/D/G spaces; it is not a physical validation
% of the four-piece bolted Abaqus column.
%
% Usage:
%   cd('D:\\CFS-Column\\git_hub')
%   benchmark_fcfsm_classifier_cufsm
% or
%   benchmark_fcfsm_classifier_cufsm('D:\\temp\\cufsm_fcfsm_reference.json')

if nargin < 1 || isempty(output_path)
    output_path = fullfile(pwd, 'cufsm_fcfsm_reference.json');
end

repo_root = fileparts(mfilename('fullpath'));
cufsm_root = fullfile(repo_root, 'cufsm-git-5.70');
if ~isfolder(cufsm_root)
    error('CUFSM 5.70 source folder not found: %s', cufsm_root);
end
addpath(genpath(cufsm_root));

required = {'SecAnal_fcFSM','klocal','trans','assemble','elemprop'};
for i = 1:numel(required)
    if exist(required{i}, 'file') ~= 2
        error('Required CUFSM function not found on path: %s', required{i});
    end
end

% Independent open-channel benchmark geometry; units N-mm-MPa.
E = 200000.0;
nu = 0.30;
G = E/(2*(1+nu));
t = 2.0;
member_length = 1000.0;
m = 2;
BC = 'S-S';

% Three flat plates, four nodes. Sharp vertices are folds, not finite-radius
% corner strips, so cornerStrips is empty for this benchmark.
xy = [0.0, 0.0;
      50.0, 0.0;
      50.0, 100.0;
      0.0, 100.0];
connectivity = [1,2; 2,3; 3,4];
nNode = size(xy,1);
nElem = size(connectivity,1);

node = zeros(nNode,8);
node(:,1) = (1:nNode)';
node(:,2:3) = xy;
node(:,4:7) = 1;
node(:,8) = 1.0;

elem = zeros(nElem,5);
elem(:,1) = (1:nElem)';
elem(:,2:3) = connectivity;
elem(:,4) = t;
elem(:,5) = 1;
cornerStrips = [];

% Native fcFSM geometry/force definitions.
[C_L, J_D, J_GD] = SecAnal_fcFSM(node, elem, cornerStrips);

% Assemble native CUFSM elastic K for this exact S-S harmonic using the
% unmodified CUFSM element functions.
K = sparse(zeros(4*nNode,4*nNode));
Kzero = sparse(zeros(4*nNode,4*nNode));
elprop = elemprop(node, elem, nNode, nElem);
m_a = m;
for iElem = 1:nElem
    b = elprop(iElem,2);
    alpha = elprop(iElem,3);
    k_local = klocal(E,E,nu,nu,G,t,member_length,b,BC,m_a);
    kg_local_zero = sparse(size(k_local,1), size(k_local,2));
    [k_global, kg_global_zero] = trans(alpha, k_local, kg_local_zero, m_a);
    nodei = elem(iElem,2);
    nodej = elem(iElem,3);
    [K, Kzero] = assemble(K, Kzero, k_global, kg_global_zero, ...
                          nodei, nodej, nNode, m_a);
end
K = (K + K')/2;

% Native fcFSM D and G spaces, matching stripmain_fcFSM.m.
C_D = K\J_GD*J_D;
J_G = null(C_D'*J_GD);
C_G = K\J_GD*J_G;

% Deterministic full-space probe. The same vector is projected by the Python
% implementation, so family shares are compared on identical physical DOFs.
nDof = 4*nNode;
idx = (1:nDof)';
probe = sin(0.37*idx) + 0.2*cos(0.11*idx);
C_all = [C_L, C_D, C_G];
phi = C_all\probe;
nL = size(C_L,2);
nD = size(C_D,2);
L_sub = C_L*phi(1:nL);
D_sub = C_D*phi(nL+1:nL+nD);
G_sub = C_G*phi(nL+nD+1:end);

E_all = 0.5*full(probe'*K*probe);
E_L = 0.5*full(L_sub'*K*L_sub);
E_D = 0.5*full(D_sub'*K*D_sub);
E_G = 0.5*full(G_sub'*K*G_sub);
shares = 100.0*[E_L,E_D,E_G]/E_all;
reconstruction_error = norm(L_sub + D_sub + G_sub - probe)/norm(probe);

reference = struct( ...
    'schema_version',2, ...
    'source',struct( ...
        'program','CUFSM', ...
        'version','5.70', ...
        'method','native SecAnal_fcFSM + klocal/trans/assemble', ...
        'scope','open-section implementation benchmark; not four-piece Abaqus validation'), ...
    'geometry',struct( ...
        'name','open_channel_three_plate', ...
        'nodes',xy, ...
        'elements',connectivity, ...
        'corner_element_ids',cornerStrips, ...
        'thickness_mm',t), ...
    'material',struct('E_MPa',E,'nu',nu), ...
    'boundary_conditions',struct('longitudinal',BC), ...
    'harmonic',struct('m',m,'length_mm',member_length), ...
    'probe_vector',probe, ...
    'K0',full(K), ...
    'families',struct( ...
        'L',struct('share_percent',shares(1),'basis',full(C_L)), ...
        'D',struct('share_percent',shares(2),'basis',full(C_D)), ...
        'G',struct('share_percent',shares(3),'basis',full(C_G))), ...
    'native_reconstruction_relative',reconstruction_error);

encoded = jsonencode(reference);
[fid,message] = fopen(output_path,'w');
if fid < 0
    error('Could not open output file %s: %s', output_path, message);
end
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, encoded, 'char');

fprintf('CUFSM fcFSM native reference written: %s\n', output_path);
fprintf('Native shares [L D G] = [%.12g %.12g %.12g] %%\n', ...
        shares(1),shares(2),shares(3));
fprintf('Native reconstruction relative error = %.3e\n', reconstruction_error);
end
