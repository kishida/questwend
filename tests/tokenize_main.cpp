// Standalone tokenizer tool: encodes text with the questwend tokenizer and
// prints one token id per line. Used to cross-check tokenization against
// llama.cpp (`llama-tokenize -m model.gguf --file f --ids --no-bos`).
//
//   qw-tokenize -m model.gguf [-f textfile]     (default: read stdin)
//   qw-tokenize -m model.gguf -d [-f idsfile]   decode ids (one per line) to text

#include "model.h"
#include "tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using namespace questwend;

int main(int argc, char ** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);   // no \n -> \r\n translation
    _setmode(_fileno(stdin),  _O_BINARY);
#endif
    std::string model_path, file;
    bool decode = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (a == "-f" && i + 1 < argc) file = argv[++i];
        else if (a == "-d") decode = true;
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf [-f textfile]\n", argv[0]);
        return 1;
    }
    std::string text;
    {
        FILE * in = file.empty() ? stdin : fopen(file.c_str(), "rb");
        if (!in) { fprintf(stderr, "cannot open %s\n", file.c_str()); return 1; }
        char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), in)) > 0) text.append(buf, n);
        if (!file.empty()) fclose(in);
    }
    auto model = Model::load(model_path);   // metadata only; weights stay on disk
    Tokenizer tok(model->vocab());
    if (decode) {
        std::string out;
        size_t p = 0;
        while (p < text.size()) {
            const int32_t id = (int32_t) strtol(text.c_str() + p, nullptr, 10);
            out += tok.decode(id);
            while (p < text.size() && text[p] != '\n') ++p;
            ++p;
        }
        fwrite(out.data(), 1, out.size(), stdout);
    } else {
        for (int32_t id : tok.encode(text, /*add_bos=*/false)) printf("%d\n", id);
    }
    return 0;
}
