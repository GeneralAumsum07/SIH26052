# usage: sweep.sh <split dir under /mnt/c/.../SIH_2026/data> <out tsv>
cd "$(dirname "$0")"
gcc -g -O2 -include math.h -w harness_wav.c pesqmod_flag.c pesqdsp.c dsp.c -lm -o hw_flag
gcc -g -O1 -fno-omit-frame-pointer -fsanitize=address -include math.h -w harness_wav.c pesqmod_flag.c pesqdsp.c dsp.c -lm -o hw_asan
R=/mnt/c/Users/Rachit/Desktop/Projects/SIH_2026/data/$1
one() { c="${1%.mix.wav}.clean.wav"; id="$(basename "$(dirname "$1")")/$(basename "$1" .mix.wav)";
  f=$(./hw_flag "$c" "$1" "$id" 2>&1 | tail -1); ASAN_OPTIONS=detect_leaks=0 ./hw_asan "$c" "$1" "$id" >/dev/null 2>asan_$$.txt; rc=$?;
  s=$(grep -o 'SUMMARY.*' asan_$$.txt | sed 's#/mnt/[^ ]*/##' | head -1); echo "$f	asan_rc $rc	$s"; }
export -f one
find "$R" -name '*.mix.wav' ! -name '*.twin.mix.wav' | sort | xargs -P 2 -I{} bash -c 'one "{}"' > "$2"
echo SWEEP_DONE $(wc -l < "$2")
