import * as rustModule from "./pkg/rust_snake.js";

console.log("TEST3")

// Create a new canvas element
let canvas = document.createElement('canvas');
canvas.width = 400;
canvas.height = 400;
let content = document.getElementById('content');
content.appendChild(canvas);

// basic game step and timing
let STEP_LENGTH = 100
let start;

function step(timeStamp) {
  if (start === undefined) {
    start = timeStamp;
  }
  const elapsed = timeStamp - start;
  if (elapsed > STEP_LENGTH) {
    start = timeStamp;
    game.step(ctx, key);
    if (game.get_gameover()) {
      alert("You died. Play again?");
      game = rustModule.Game.new(ctx, 20, 20, 20);

    }
    window.requestAnimationFrame(step);    
  }
  else {
    window.requestAnimationFrame(step);
  }
}

// animation stuff
let ctx = canvas.getContext("2d");

// import wasm  
import init from "./pkg/rust_snake.js";
await init().then(() => {
  rustModule.print_obj("Imported wasm successfully.");
});

let game = rustModule.Game.new(ctx, 20, 20, 20);

let key = "ArrowUp";
document.addEventListener('keydown', function(event) {
  if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) {
    key = event.key;
  }
});


window.requestAnimationFrame(step);