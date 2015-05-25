using Word2Vec
using Base.Test

data_dir = joinpath(Pkg.dir("Word2Vec"), "test", "data")
test_file = joinpath(data_dir, "text8_tiny")

printover(i) = print("\r$i\e[K")

function test_word_window()
    open(test_file) do fp
        test_word_window(fp)
    end
end

function test_word_window(fp)
    println("Testing words_of...")

    for i in Word2Vec.words_of(fp)
        printover(i)
    end

    seekstart(fp)
    for i in map(uppercase, Word2Vec.words_of(fp))
        printover(i)
    end

    println("\rDone.\e[K")

    println("Testing sliding_window...")
    seekstart(fp)
    for i in Word2Vec.sliding_window(Word2Vec.words_of(fp))
        printover(join(i, ','))
    end
    println("\nDone.")
end

test_word_window()
