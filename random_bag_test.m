nPos = 1000;
nNeg = 1000;
nPosBag =100;
nNegBag = 100;

posIns = randi(floor(nPos/nPosBag),nPosBag,1);
while sum(posIns)~=nPos
    ind_add_to = randi(length(posIns));
    posIns(ind_add_to) = posIns(ind_add_to)+randi(nPos-sum(posIns));
end


negIns = randi(floor(nNeg/(nPosBag+nNegBag))...
                        ,(nPosBag+nNegBag),1);
while sum(negIns)~=nNeg
    ind_add_to = randi(length(negIns))
    negIns(ind_add_to) = negIns(ind_add_to)+randi(nNeg-sum(negIns));
end

PosNegIns = [[posIns,negIns(1:nPosBag)];...
                        [zeros(nNegBag,1),negIns(nPosBag+1:end)]];