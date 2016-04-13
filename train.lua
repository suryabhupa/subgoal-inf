require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'xlua'
require 'image'

noutputs = 50

nfeats = 3 -- because image is RGB
width = 210
height = 320
ninputs = nfeats*width*height

nstates = {64, 64, 128}
dims = {49, 77}
filtsize = 5
poolsize = 2 

function create_model()
  --  CNN
	local convnet = nn.Sequential()
	convnet:add(nn.SpatialConvolution(nfeats, nstates[1], filtsize, filtsize))
	convnet:add(nn.ReLU())
	convnet:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

	convnet:add(nn.SpatialConvolution(nstates[1], nstates[2], filtsize, filtsize))
	convnet:add(nn.ReLU())
	convnet:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

	convnet:add(nn.View(nstates[2]*dims[1]*dims[2]))

	convnet:add(nn.Dropout(0.5))
	convnet:add(nn.Linear(nstates[2]*dims[1]*dims[2], nstates[3]))
	convnet:add(nn.ReLU())
	convnet:add(nn.Linear(nstates[3], noutputs)) 

  -- One-hot encoding
	local onehot = nn.Sequential()
	onehot:add(nn.Linear(9, 9)) 

  -- ParallelTable 
	local parallel = nn.ParallelTable()
	parallel:add(onehot)
	parallel:add(convnet)

  nc = 3
  dstates = 100
  genfilters = 64

  -- Deconvolution
  local decnet = nn.Sequential()
  -- input to convolution
  decnet:add(nn.SpatialFullConvolution(dstates, genfilters * 8, 4, 4))
  decnet:add(nn.SpatialBatchNormalization(genfilters * 8)):add(ReLU(true))
  -- state size: (genfilters*8) x 4 x 4
  decnet:add(nn.SpatialFullConvolution(genfilters * 8, genfilters * 4, 4, 4, 2, 2, 1, 1))
  decnet:add(nn.SpatialBatchNormalization(genfilters * 4)):add(nn.ReLU(true))
  -- state size: (genfilters*4) x 8 x 8
  decnet:add(nn.SpatialFullConvolution(genfilters * 4, genfilters * 2, 4, 4, 2, 2, 1, 1))
  decnet:add(nn.SpatialBatchNormalization(genfilters * 2)):add(nn.ReLU(true))
  -- state size: (genfilters*2) x 16 x 16
  decnet:add(nn.SpatialFullConvolution(genfilters * 2, genfilters, 4, 4, 2, 2, 1, 1))
  decnet:add(nn.SpatialBatchNormalization(genfilters)):add(nn.ReLU(true))
  -- state size: (genfilters) x 32 x 32
  decnet:add(nn.SpatialFullConvolution(genfilters, nc, 4, 4, 2, 2, 1, 1))
  decnet:add(nn.Tanh())

  --TODO: connect back into an image

	print('==> creating model')
	local model = nn.Sequential()
	model:add(parallel)
	model:add(nn.JoinTable(1))
  model:add(decnet)

	-- Add more layers here for deconvolution
	print(model)
	return model

end

function define_criterion()
	print('==> define loss')
end

function get_action_data(file)
	-- read in the acations in the file, line by line
	data = {}
	io.input(file)
	while true do
		local line = io.read()
		if line == nil then break end
			table.insert(data, tonumber(line))
	end
	return data
end

function get_frame_data(dir)
	-- get all *.png files in the directory and sort them
	files = {}
	for file in paths.files(dir) do
		if file:find("png$") then
			table.insert(files, paths.concat(dir, file))
		end
	end 
	table.sort(files, function (a,b) return a < b end) 
	
	return files
end

function make_training_data(action_data_file, frame_data_dir)
	a_data = get_action_data(action_data_file)
	f_data = get_frame_data(frame_data_dir)

	print(a_data)
	print(f_data)
	return {a_data, f_data}
end

function test(model)
end

function main()
	model = create_model()
	-- training of autoencoder here
	training_data = make_training_data("ALE/doc/examples/record/game_actions.txt", "ALE/doc/examples/record/")

	model:training() -- put into training mode (dropout turns on)

	i = image.load("atari/exp1/000000.png")
	j = torch.rand(9)

  -- TODO: put training of model somewhere here
	print(model:forward{j, i})
	print(i:size())
end

main()

--[[
if opt.visualize then
   if opt.model == 'convnet' then
      if itorch then
	 print '==> visualizing ConvNet filters'
	 print('Layer 1 filters:')
	 itorch.image(model:get(1).weight)
	 print('Layer 2 filters:')
	 itorch.image(model:get(5).weight)
      else
	 print '==> To visualize filters, start the script in itorch notebook'
      end
   end
end
--]]

