# [TensorFlow.NET](https://github.com/SciSharp/TensorFlow.NET) (C#)

(To set up C# to run in VSCode: https://github.com/hchiam/learning-csharp)

https://github.com/SciSharp/TensorFlow.NET

```sh
# to set up a new C# project once:
dotnet new console --framework net7.0
# to run:
dotnet run
```

```sh
dotnet add package TensorFlow.NET
dotnet add package TensorFlow.Keras
# install TensorFlow binary:
  # CPU version
dotnet add package SciSharp.TensorFlow.Redist
  # GPU version (requires CUDA and cuDNN)
# dotnet add package SciSharp.TensorFlow.Redist-Windows-GPU
```

https://scisharp.github.io/tensorflow-net-docs/#/essentials/introduction?id=c-example

https://github.com/SciSharp/SciSharp-Stack-Examples/blob/master/src/TensorFlowNET.Examples/HelloWorld.cs
