# małe funkcje pomocnicze

# zwraca indeksy klas o największym logitach
function argmax_classes(logits)
    if ndims(logits) == 1
        return [argmax(logits)]
    elseif ndims(logits) == 2
        return vec(map(argmax, eachcol(logits))) # argmax dla każdej kolumny batcha
    else
        error("argmax_classes supports 1D or 2D tensors")
    end
end

# porównanie indeksów klas o największych logitach z indeksami klas w etykietach
function accuracy(logits, labels)
    preds = argmax_classes(logits)
    truth = argmax_classes(labels)
    return sum(preds .== truth) / length(preds) # dokładność jako odsetek poprawnych predykcji
end