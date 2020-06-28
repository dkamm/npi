import argparse
import curses
import json
import time


from npi.task import addition

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('examplepath', type=str)
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()

    examples = json.load(open(args.examplepath))
    example = examples[args.idx]

    frames = []
    if example['task'] == 'addition':
        env = addition.Environment(*example['inputs'])
        for subroutine, args in example['trace']:
            if subroutine == addition.Subroutine.ACT:
                env.step(*args)
                argsstr = ' '.join(env.decode_arguments(*args))
            else:
                argsstr = ' '
            frames.append(
                'Subroutine: {}\n'.format(addition.Subroutine(subroutine).name) +
                'Args: {}\n'.format(argsstr) +
                env.render()
            )

    def play_frames(stdscr):
        for frame in frames:
            stdscr.addstr(0, 0, frame)
            stdscr.refresh()
            time.sleep(.4)
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()

    curses.wrapper(play_frames)
    print(frames[-1])


if __name__ == '__main__':
    main()
