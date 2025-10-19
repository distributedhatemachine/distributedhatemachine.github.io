---
title: "Reviving Jetson Nano to run some LMs and kernels"
author: "Egor"
date: "2025-10-19"
summary: "yay cuda kernels!!"
description: "yay cuda kernels!!"
toc: true
readTime: true
autonumber: true
math: true
tags: ["cuda", "jetson"]
showTags: false
hideBackToTop: false
---


Seven years ago I bought said Jetson Nano 2gb for 70 bucks. I did run some graphics and CV examples, but I mostly used it as multimedia server (I used to watch YouTube and films on XBox 360, so it was less weird alternative).  
Today is someday in october 2025 and I want to bring this beast with whopping 128 CUDA cores and see, can it survive "The Trial of the LLMs".

## Setup

First of all we need to get Linux running on machine. MicroSD card I bought with Jetson was used for my DSLR, so let's build image from scratch.  
I still remember that official NVidia images are pain in a ass, fortunately we are blessed with better (and lighter!) alternative [pythops/jetson-image](https://github.com/pythops/jetson-image).

I've hid commands so they don't take so much space, but providing them is essential, because some of them are not trivial.

{{< details "For some reason I couldn't run it as is and had to use Multipass to create Ubuntu VM" >}} 
```
brew install --cask multipass
multipass launch --name builder --cpus 4 --mem 8G --disk 50G jammy
multipass shell builder
```
{{< /details >}}

{{< details "Get Prebuilt MPR" >}} 
```
wget -qO - 'https://proget.makedeb.org/debian-feeds/prebuilt-mpr.pub' | gpg --dearmor | sudo tee /usr/share/keyrings/prebuilt-mpr-archive-keyring.gpg 1> /dev/null
echo "deb [arch=all,$(dpkg --print-architecture) signed-by=/usr/share/keyrings/prebuilt-mpr-archive-keyring.gpg] https://proget.makedeb.org prebuilt-mpr $(lsb_release -cs)" | sudo tee /etc/apt/sources.list.d/prebuilt-mpr.list
sudo apt update
```
{{< /details >}}

{{< details "Installing said tools" >}} 
```
sudo apt install -y git just jq podman qemu-user-static
```
{{< /details>}}

After setup we can 
```
git clone https://github.com/pythops/jetson-image
cd jetson-image
```
and make rootfs with 5 major versions old Ubuntu
```
just build-jetson-rootfs 20.04
```
build image for our board
```
just build-jetson-image -b jetson-nano-2gb -l 32
```
copy image back to host
```
multipass transfer builder:/home/ubuntu/jetson-image/jetson.img .
```
and finally flash image into board
```
sudo dd if=jetson.img of=/dev/diskN bs=1m
```


## Logging in
After typing login `jetson` and password `jetson` we are in!  
Running `free -m` shows a total of `~1700 mb` of 64-bit LPDDR4 to spare. That's CPU RAM and GPU VRAM **combined**, so we'll try to use most of it.

Obviously engines as `vLLM` or `sglang` are out of consideration so we have to resort to [llama.cpp](https://github.com/ggml-org/llama.cpp). We could write something from scratch using minimal overhead like [llama.c](https://github.com/karpathy/llama2.c), but I want to run modern LLMs and not debug C/Cpp/Rust code.

Here is first pitfall. `llama.cpp` doesn't provide `linux-cuda-arm64-smth` binary, so building it from sources is only option.

