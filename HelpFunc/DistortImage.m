function [XOut, YOut] = DistortImage(XIn, YIn)

GS = InitializeGS(XIn);

XRotPos = TransformImage(XIn, GS, 'Rotation', pi/6);
XRotNeg = TransformImage(XIn, GS, 'Rotation', -pi/6);
XScaleX = TransformImage(XIn, GS, 'Scale', [1.1 1]);
XScaleY = TransformImage(XIn, GS, 'Scale', [1 1.1]);

XOut = [XRotPos XRotNeg XScaleX XScaleY];
YOut = [YIn YIn YIn YIn];

function XT = TransformImage(XI, GS, Type, Var)

switch Type
    case 'Rotation'
        T = [cos(Var) -sin(Var); sin(Var) cos(Var)];
    case 'Scale'
        T = [Var(1) 0; 0 Var(2)];
    otherwise
        T = eye(2);
end

XT = zeros(GS.Pixels, GS.DataSize);
for iData = 1:GS.DataSize
    
    ImageI = reshape(XI(:, iData), GS.PixelM, GS.PixelM); % Initial frame
    ImageT = zeros(GS.PixelM, GS.PixelM); % Transformed frame
    
    for ixGI = 1:GS.PixelM
        for iyGI = 1:GS.PixelM
            
            PixelIntensity = ImageI(iyGI, ixGI);
            if PixelIntensity > 0
                
                rCI = [GS.xCV(ixGI) GS.yCV(iyGI)]'; % Initial position
                rCT = T * rCI; % Transform position
                
                [rGT, Rat] = rC2rG(rCT, GS); % Translate rCT to rGT
                
                for irGT = 1:size(rGT, 2)
                    ixGT = rGT(1, irGT);
                    iyGT = rGT(2, irGT);
                    iRat = Rat(irGT);
                    ImageT(iyGT, ixGT) = iRat * PixelIntensity + ...
                        ImageT(iyGT, ixGT);
                end
            end
        end
    end
    
    % Normalize & store.
    ImageT = ImageT / max(max(ImageT));
    XT(:, iData) = reshape(ImageT, [], 1);
end

function GS = InitializeGS(X)
GS = struct();

GS.DataSize = size(X, 2);
GS.Pixels   = size(X, 1);
GS.PixelM   = sqrt(GS.Pixels);

% Vector with grid (G) & cartesian (C) coordinates
GS.xCV = -(GS.PixelM - 1):2:(GS.PixelM - 1);
GS.yCV = (GS.PixelM - 1):-2:-(GS.PixelM - 1);
GS.xGV = 1:GS.PixelM;
GS.yGV = 1:GS.PixelM;

% Translation parameters k and m (C to G)
GS.krG = 0.5 * [1 -1]';
GS.mrG = (GS.PixelM + 1)/2 * [1 1]';

function [rG, Rat] = rC2rG(rCRef, GS)

% Translate rCRef to rGRef
rGRef = GS.krG.*rCRef + GS.mrG;

% Find closest grids
xGSearch = max(round(rGRef(1) - 1), 1):min(round(rGRef(1) + 1), GS.PixelM);
yGSearch = max(round(rGRef(2) - 1), 1):min(round(rGRef(2) + 1), GS.PixelM);
rG       = [];
for ixG = 1:length(xGSearch)
    for iyG = 1:length(yGSearch)
        xGTmp = xGSearch(ixG);
        yGTmp = yGSearch(iyG);
        rGTmp = [xGTmp yGTmp]';
        Delta = (rGTmp - rGRef);
        if (Delta(1)^2 + Delta(2)^2) < 1
            rG = [rG rGTmp];
        end
    end
end

% Compute ratio
Rat = zeros(1, size(rG, 2));
for Idx = 1:size(rG, 2)
    DistX    = 1 - abs(rG(1, Idx) - rGRef(1));
    DistY    = 1 - abs(rG(2, Idx) - rGRef(2));
    Rat(Idx) = DistX*DistY;
end
