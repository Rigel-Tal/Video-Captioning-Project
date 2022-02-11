import torch
from torchtext.data.metrics import bleu_score

def translate(model, vid, dataset, device, max_length=50):
    # Build encoder hidden, cell state
    with torch.no_grad():
        encoder_state, hidden, cell = model.encoder(vid)

    outputs = [dataset.vocab.stoi["<SOS>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, encoder_state, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == dataset.vocab.stoi["<EOS>"]:
            break

    translated_sentence = [dataset.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]


def bleu(data, model, dataset, device):
    targets = []
    outputs = []

    for idx, (vid, captions) in enumerate(data):
        src = vid.to(device)
        trg = captions.to(device)
        trg = trg.squeeze(1)

        prediction = translate(model, src, dataset, device)
        prediction = prediction[:-1]  # remove <EOS> token

        target = [dataset.vocab.itos[int(idx.item())] for idx in trg]
        targets.append(target)
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    print("Done")


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
