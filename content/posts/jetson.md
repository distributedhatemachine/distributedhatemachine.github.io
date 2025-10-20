---
title: "Reviving Jetson Nano to run some LLMs"
author: "Egor"
date: "2025-10-19"
summary: "Large Language Models—minus the Large—on a tiny Jetson Nano."
description: "Large Language Models—minus the Large—on a tiny Jetson Nano."
toc: true
readTime: true
autonumber: true
math: true
tags: ["cuda", "jetson"]
showTags: false
hideBackToTop: false
---


Seven years ago I bought a Jetson Nano 2GB for $70. I ran some graphics and computer vision examples, but I mostly used it as a multimedia server (I used to watch YouTube and films on an Xbox 360, so it was a less weird alternative).  
Today is someday in October 2025, and I want to bring this beast, with a whopping 128 CUDA cores, back to life and see whether it can survive "The Trial of the LLMs".

## Setup

First of all, we need to get Linux running on the machine. The microSD card I bought with the Jetson was used for my DSLR, so let's build an image from scratch.  
I still remember that the official NVIDIA images are a pain; fortunately, we are blessed with a better (and lighter!) alternative: [pythops/jetson-image](https://github.com/pythops/jetson-image).

{{< details "I've hidden the commands so they don't take up so much space, but providing them is essential because some of them are not trivial." >}}

- For some reason I couldn't run it as-is and had to use Multipass to create an Ubuntu VM
```
brew install --cask multipass
multipass launch --name builder --cpus 4 --mem 8G --disk 50G jammy
multipass shell builder
```

- Get Prebuilt MPR
```
wget -qO - 'https://proget.makedeb.org/debian-feeds/prebuilt-mpr.pub' | gpg --dearmor | sudo tee /usr/share/keyrings/prebuilt-mpr-archive-keyring.gpg 1> /dev/null
echo "deb [arch=all,$(dpkg --print-architecture) signed-by=/usr/share/keyrings/prebuilt-mpr-archive-keyring.gpg] https://proget.makedeb.org prebuilt-mpr $(lsb_release -cs)" | sudo tee /etc/apt/sources.list.d/prebuilt-mpr.list
sudo apt update
```

- Install the some tools
```
sudo apt install -y git just jq podman qemu-user-static
```
{{< /details >}}

After setup, we can 
```
git clone https://github.com/pythops/jetson-image
cd jetson-image
```
and make a rootfs with a five-major-versions-old Ubuntu
```
just build-jetson-rootfs 20.04
```
Build the image for our board
```
just build-jetson-image -b jetson-nano-2gb -l 32
```
Copy the image back to the host
```
multipass transfer builder:/home/ubuntu/jetson-image/jetson.img .
```
and finally flash the image onto the board
```
sudo dd if=jetson.img of=/dev/diskN bs=1m
```

{{< details "Setting up WiFi" >}}
I thought that this would be as easy as
```
sudo nmcli device wifi list
sudo nmcli device wifi connect "YourSSID" password "YourPassword"
```
But `pythops/jetson-image` is too lightweight — even `iwconfig` or any networking doesn't work.  
I forked the repo and added some files to support networking out of the box — [wtfnukee/jetson-image](https://github.com/wtfnukee/jetson-image.git). It installs the missing packages so the commands above work.
{{< /details >}}


## Logging in
After typing the login `jetson` and password `jetson`, we're in!  
Running `free -m` shows a total of `~1700 MB` of 64-bit LPDDR4 to spare. That's CPU RAM and GPU VRAM **combined**, so we'll try to use as much of it as possible.

Obviously, beefy engines like `vLLM` or `sglang` are out of the question, so we have to resort to [llama.cpp](https://github.com/ggml-org/llama.cpp). We could write something from scratch using minimal overhead like [llama.c](https://github.com/karpathy/llama2.c), but I want to run modern LLMs and not debug C/C++/Rust code.

Here is the first pitfall: `llama.cpp` doesn't provide a `linux-cuda-arm64-smth` binary, so building it from source is the only option.  
Some [madlad](https://www.caplaz.com/jetson-nano-running-llama-cpp/) already did this, but they had the 4GB version, it will be interesting to see speed difference.

## Benchmarking
I thought that just running couple of prompt is too simple, so inspired by [LLM's Engineer Almanac by Modal](https://modal.com/llm-almanac/how-to-benchmark) I've decided to write my own benchmark.

Original code [modal-labs/stopwatch](https://github.com/modal-labs/stopwatch) runs on Modal and tested vLLM, SGLang, and TensorRT-LLM.  
My version [wtfnukee/hourglass](https://github.com/wtfnukee/hourglass) is smaller (and probably slower), hence the name. It uses `llama-bench` suite for now, but I left foundation for handwritten benchmark engine. I'll cover this tool and benchmarking as a topic in separate post.

TO BE CONTINUED...

## Acknowledgments
-[llama.cpp guide](https://blog.steelph0enix.dev/posts/llama-cpp-guide/#llama-bench) by steelph0enix
- [LLM's Engineer Almanac](https://modal.com/llm-almanac/how-to-benchmark) by Modal (pls hire me)
- pythops for [jetson-image](https://github.com/pythops/jetson-image) and [tegrastats](https://github.com/pythops/tegrastats)